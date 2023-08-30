import torch
import transformers
from transformers import RobertaTokenizerFast


def load_SecureBERT(model_name="/media/ea/SSD2/Projects/CVE2TTP/models/SecureBERT", mode="mlm"):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = None
    if mode == "mlm":
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name)
    elif mode == "clm":
        model = transformers.RobertaForCausalLM.from_pretrained(model_name)

    return tokenizer, model
def predict_mask(sent, tokenizer, model, topk=10, print_results=True):
    token_ids = tokenizer.encode(sent, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]
    words = []
    with torch.no_grad():
        output = model(token_ids)

    last_hidden_state = output[0].squeeze()

    list_of_list = []
    for index, mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=topk, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        words = [w.replace(' ', '') for w in words]
        list_of_list.append(words)
        if print_results:
            print("Mask ", "Predictions : ", words)

    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess + "," + j[0]

    return words

def generate_text(model, tokenizer, prompt="Once upon a time",max_length=100, num_return_sequences=5):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"{prompt} {generated_text}")
    return generated_text