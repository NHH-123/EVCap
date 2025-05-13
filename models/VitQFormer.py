import logging
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import random
from models.blip2 import Blip2Base, disabled_train
from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import pickle
import faiss
import re
from collections import Counter
import spacy



class EVCap(Blip2Base):
    
    def __init__(
        self,
        ext_path,
        caption_ext_path,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model="",
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=160,
        end_sym='\n',
        low_resource=False,
        device_8bit=0,
    ):
        super().__init__()

        self.low_resource = low_resource
        self.topn = topn
        print('topn:', self.topn)

        ##### Image 
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = True
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')


        ##### Text 
        self.bert_tokenizer = self.init_tokenizer()
        self.Qformer_txt, self.query_tokens_txt = self.init_Qformer_txt(
            num_query_token_txt, self.Qformer.config.hidden_size
        )
        self.Qformer_txt.resize_token_embeddings(len(self.bert_tokenizer))
        self.Qformer_txt.cls = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        if freeze_qformer:
            for name, param in self.Qformer_txt.named_parameters():
                param.requires_grad = False
            self.Qformer_txt = self.Qformer_txt.eval()
            self.Qformer_txt.train = disabled_train
            self.query_tokens_txt.requires_grad = True
            logging.info("freeze Qformer")
        print('query_tokens_txt', self.query_tokens_txt.shape)
        print('Loading Q-Former Done')
        print('Loading Q-Former_txt Done')


        

        # self.max_txt_len = max_txt_len
        # self.end_sym = end_sym

        # if prompt_path:
        #     with open(prompt_path, 'r') as f:
        #         raw_prompts = f.read().splitlines()
        #     filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
        #     self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
        #     print('Load {} training prompts'.format(len(self.prompt_list)))
        #     print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        # else:
        #     self.prompt_list = []
        # print(ext_path)
        # with open(ext_path, 'rb') as f:
        #     ext_base_img, self.ext_base_img_id = pickle.load(f)
        #     print(ext_base_img.shape, len(self.ext_base_img_id))
        #     feature_library_cpu = ext_base_img.cpu().numpy()
        #     faiss.normalize_L2(feature_library_cpu)
        #     self.feat_index = faiss.IndexFlatIP(feature_library_cpu.shape[1])
        #     self.feat_index.add(feature_library_cpu)
        #     print(f"loaded external base image")
        # self.nlp = spacy.load("en_core_web_sm")
        # print("loaded spacy")
        # with open(caption_ext_path,'rb') as f:
        #     caption_ext_base_img, self.caption_ext_base_img_id = pickle.load(f)
        #     print(caption_ext_base_img.shape,len(self.caption_ext_base_img_id))
        #     caption_feature_library_cpu = ext_base_img.cpu().numpy()
        #     faiss.normalize_L2(caption_feature_library_cpu)
        #     self.caption_feat_index = faiss.IndexFlatIP(caption_feature_library_cpu.shape[1])
        #     self.caption_feat_index.add(caption_feature_library_cpu)
        #     print(f"loaded caption external base image")


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()
    
    def get_img_features(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) #(B,T,encoder_hidden_states)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) #(B,T)

            #self.query_tokens = (1,num_query_tokens=32,encoder_hidden_size) expands to (B,num_query_tokens=32,encoder_hidden_size)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state #(B,num_query_tokens=32,Q_former_hidden_size=768)
            query = torch.mean(query_output_img,dim=1) #average over num_query_tokens => (B,1,768)
            return query


    def prompt_wrap(self, img_embeds, atts_img, prompt_list):
        if prompt_list:
            batch_size = img_embeds.shape[0]
            emb_lists = []
            for i in range(batch_size):
                prompt = random.choice(prompt_list)
                p_before, p_after = prompt.split("<ImageHere>", 1)
                self.llama_tokenizer.padding_side = "right"
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)        
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
                img_embeds_i = img_embeds[i].unsqueeze(0)
                wrapped_embed_i = torch.cat([p_before_embeds, img_embeds_i, p_after_embeds], dim=1)
                emb_lists.append(wrapped_embed_i)  

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.llama_model.model.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img


    def pre_name(self, caption):
        caption = re.sub(
            r"([_!,'\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption

    def retrieve_similar_features(self, query_features, feat_index, image_id, top_k = 5, sub_top_k = 32):
        batch_size, nums, dims = query_features.shape #(B,num_query_tokens,encoder_hidden_states)
        query_features = query_features.view(-1,dims)   #(B*num_query_tokens,encoder_hidden_states)

        query_features_cpu = query_features.detach().cpu().numpy()
        faiss.normalize_L2(query_features_cpu)
        top_k_similarities, top_k_indices = feat_index.search(query_features_cpu, top_k) #Both size (B*num_query_tokens,top_k)

        top_k_indices = torch.tensor(top_k_indices).to(device = query_features.device) #(B*num_query_tokens,top_k)
        top_k_similarities = torch.tensor(top_k_similarities).to(device = query_features.device) 
        top_k_similarities = top_k_similarities.view(batch_size, -1) #(B,num_query_tokens*top_k)

        indices = top_k_indices.view(batch_size, -1) #(B,num_query_tokens*top_k)

        re_txt_list_all = []    
        for batch_i in range(batch_size):
            indices_list = indices[batch_i]
            re_txt_batch_list = []
            for i in indices_list: 
                re_txt_batch_list.append(image_id[i])
            re_txt_list_all.append(re_txt_batch_list) #(B,num_query_tokens*top_k)
         
        sorted_batched_ret = []
        for listA, listB in zip(top_k_similarities, re_txt_list_all):
            sorted_listA, indices = listA.sort(descending=True)
            sorted_listB = [self.pre_name(listB[idx]) for idx in indices]
            sorted_listB = sorted_listB[:sub_top_k]
            sorted_batched_ret.append(sorted_listB)
        return sorted_batched_ret  #(B,sub_top_k)
    
    def extract_objects_actions(self,caption):
    
        doc = self.nlp(caption)
        objects = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
        actions = [token.text for token in doc if token.pos_ == "VERB"]
        return objects, actions

    def retrieve_caption_and_filter(self,feat_index, image_id, query_features, top_n, top_k=5, sub_top_k=32):
        """
        Retrieve captions based on FAISS search results, extract objects and actions, and arrange them into batches.

        Parameters:
            feat_index (faiss.Index): The FAISS index.
            image_id (list): List of image IDs corresponding to the features in the index.
            caption_base_id (dict): A dictionary mapping image IDs to captions.
            query_features (torch.Tensor): Query features for retrieval (shape: (B, num_query_tokens, encoder_hidden_size)).
            top_k (int): Number of top similar results to retrieve.
            sub_top_k (int): Number of captions to process for object/action extraction.

        Returns:
            dict: A dictionary containing batched objects and actions.
        """
        batch_size, nums, dims = query_features.shape #(B,num_query_tokens,encoder_hidden_states)
        query_features = query_features.view(-1,dims)   #(B*num_query_tokens,encoder_hidden_states)

        query_features_cpu = query_features.detach().cpu().numpy()
        faiss.normalize_L2(query_features_cpu)
        top_k_similarities, top_k_indices = feat_index.search(query_features_cpu, top_k) #Both size (B*num_query_tokens,top_k)

        top_k_indices = torch.tensor(top_k_indices).to(device = query_features.device) #(B*num_query_tokens,top_k)
        top_k_similarities = torch.tensor(top_k_similarities).to(device = query_features.device) 
        top_k_similarities = top_k_similarities.view(batch_size, -1) #(B,num_query_tokens*top_k)

        indices = top_k_indices.view(batch_size, -1) #(B,num_query_tokens*top_k)

        retrieved_captions = []    
        for batch_i in range(batch_size):
            indices_list = indices[batch_i]
            re_txt_batch_list = []
            for i in indices_list: 
                re_txt_batch_list.append(image_id[i])
            retrieved_captions.append(re_txt_batch_list) #(B,num_query_tokens*top_k)
        
        sorted_batched_ret = []
        for listA, listB in zip(top_k_similarities, retrieved_captions):
            sorted_listA, indices = listA.sort(descending=True)
            sorted_listB = [listB[idx] for idx in indices]
            sorted_listB = sorted_listB[:sub_top_k]
            sorted_batched_ret.append(sorted_listB) #(B,sub_top_k) (B,32)
        # retrieved_captions = [
        #     "A group of women are trying to push the table to the corner of the room.",
        #     "The cat is sleeping on the mat.",
        #     "A dog runs across the park chasing a ball.",
        #     "A man is sitting on a chair reading a book.",
        #     "The table is covered with a red cloth.",
        #     "A cat is sitting on a windowsill.",
        #     "The dog jumps over the fence.",
        #     "A woman is painting a canvas in a studio.",
        #     "Children are playing with a ball in the garden.",
        #     "A person is eating a sandwich in the park."
        # ]

        # Step 4: Extract objects and actions
        batched_objects = []
        batched_actions = []

        for captions in sorted_batched_ret:
            object_counter = Counter()
            action_counter = Counter()

            for caption in captions:
                objects, actions = extract_objects_actions(caption)
                object_counter.update(objects)
                action_counter.update(actions)

            # Get top-n frequent objects and actions
            top_objects = [term for term, _ in object_counter.most_common(top_n)]
            top_actions = [term for term, _ in action_counter.most_common(top_n)]

            batched_objects.append(top_objects) #(B,num_obj)
            batched_actions.append(top_actions) #(B,num_act)

        # Step 5: Organize results into batches
        return {
            "objects": batched_objects,  # List of lists of objects, one per batch
            "actions": batched_actions  # List of lists of actions, one per batch
        }


    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) #(B,T,encoder_hidden_states)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) #(B,T)

            #self.query_tokens = (1,num_query_tokens=32,encoder_hidden_size) expands to (B,num_query_tokens=32,encoder_hidden_size)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state #(B,num_query_tokens=32,Q_former_hidden_size=768)
            query_output_img_atts = torch.ones(query_output_img.size()[:-1], dtype=torch.long).to(device) #(B,32) ?
            re_txt_list_all  = self.retrieve_similar_features(query_output_img, self.feat_index, self.ext_base_img_id)
            re_obj_act_all = self.retrieve_caption_and_filter(query_output_img,self.caption_feat_index, self.caption_ext_base_img_id)
            obj_list = re_obj_act_all["object"]
            action_list = re_obj_act_all["action"]
            re_txt_list_all = torch.cat((re_txt_list_all,obj_list),dim=-1)
            re_txt_list_batch = []
            for sublist in re_txt_list_all:
                sublist_new = []
                for item in sublist:
                    if item not in sublist_new:
                        sublist_new.append(item)
                        if len(sublist_new)>self.topn: 
                            break
                sublist_new = [" Object: "] + sublist_new
                re_txt_list_batch.append(" [SEP] ".join(sublist_new))

            re_act_list_batch = []
            for sublist in action_list:
                sublist_new = []
                for item in sublist:
                    if item not in sublist_new:
                        sublist_new.append(item)
                        if len(sublist_new)>self.topn: 
                            break
                sublist_new = [" Action: "] + sublist_new
                re_act_list_batch.append(" [SEP] ".join(sublist_new))
            
            re_final = torch.cat((re_txt_list_batch,re_act_list_batch),dim=-1)
            

            # print(re_txt_list_batch)

            

            text = self.bert_tokenizer(
                    re_final,
                    truncation=True,
                    padding="longest",
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

            #(1,num_query_tokens=32,encoder_hidden_size) expands to (B,num_query_tokens=32,encoder_hidden_size)
            query_tokens_txt = self.query_tokens_txt.expand(image_embeds.shape[0], -1, -1)
            query_atts_txt = torch.ones(query_tokens_txt.size()[:-1], dtype=torch.long).to(   #(B,num_query_tokens=32)
                image_embeds.device
            )

            query_output_img_atts = torch.ones(query_output_img.size()[:-1], dtype=torch.long).to(device)
            query_output_img_atts = torch.cat([query_atts_txt, query_output_img_atts], dim=1) #(B,64)


            attention_mask = text.attention_mask
            query_outputs_txt = self.Qformer_txt.bert(
                text.input_ids,
                query_embeds=query_tokens_txt,
                attention_mask=attention_mask,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            query_output_txt = query_outputs_txt.last_hidden_state[:, : query_tokens_txt.size(1), :]

            query_output_all = torch.cat([query_output_img, query_output_txt], dim=1) 
            qform_all_proj = self.llama_proj(query_output_all)
            atts_qform_all_proj = torch.ones(qform_all_proj.size()[:-1], dtype=torch.long).to(device)
        return qform_all_proj, atts_qform_all_proj


    def forward(self, samples):
        ##### Image
        image = samples["image"]
        qform_all_proj, atts_qform_all_proj = self.encode_img(image)
        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(qform_all_proj, atts_qform_all_proj, self.prompt_list) #(self, img_embeds, batch_names, atts_img, prompt_list):

        ##### Caption generation
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]
        text_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)


        bos = torch.ones([qform_all_proj.shape[0], 1],
                         dtype=text_tokens.input_ids.dtype,
                         device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]


        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones([qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]], 
                       dtype=torch.long).to(image.device).fill_(-100)  
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
        
        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_prompt, text_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"output": outputs[0], "loss": loss}
