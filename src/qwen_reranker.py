# reranker.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class RerankerManager:
    def __init__(self, model_name="Qwen/Qwen3-Reranker-0.6B", use_cuda=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.max_length = 8192
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.enabled = True

    def format_instruction(self, instruction, query, doc):
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def compute_logits(self, inputs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()  # "yes" 概率
        return scores

    def rerank_with_scores(self, query, doc_score_pairs, final_k=5, instruction=None):
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"

        docs = [doc.page_content for doc, _ in doc_score_pairs]
        pairs = [self.format_instruction(instruction, query, doc) for doc in docs]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        reranked = list(zip([doc for doc, _ in doc_score_pairs], [score for _, score in doc_score_pairs], scores))
        reranked = sorted(reranked, key=lambda x: x[2], reverse=True)  # 按 rerank 得分降序排列
        return reranked[:final_k]
