import random
class dataloader:
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max = max_len

    def GetData(self):
        idx = random.randint(0, len(self.df) - 1)
        return self.df.loc[idx, 'text']

    def preprocess(self):
        while True:
            full_text = self.GetData()

            # Cari batas response
            split_key = "### Response:"
            resp_index = full_text.find(split_key)

            if resp_index == -1:
                print("Warning: No response found, skipping sample.")
                continue  # skip sample

            inp_str = full_text[:resp_index + len(split_key)].strip()
            out_str = full_text[resp_index + len(split_key):].strip()

            full_concat = inp_str + "\n" + out_str

            # Tokenisasi
            tokenized = self.tokenizer(full_concat, return_tensors="pt", truncation=True, max_length=self.max)
            input_ids = tokenized.input_ids[0]
            attention_mask = tokenized.attention_mask[0]

            # Cek panjang input
            if len(input_ids) < self.max:
                break
            else:
                print("Warning: input too long, retrying...")

        # Token index dari akhir instruction untuk masking label
        start_idx = len(self.tokenizer(inp_str, return_tensors="pt").input_ids[0])

        # Siapkan label (mask bagian sebelum start_idx dengan -100)
        label_ids = input_ids.clone()
        label_ids[:start_idx] = -100

        return input_ids, label_ids, inp_str, out_str, start_idx
