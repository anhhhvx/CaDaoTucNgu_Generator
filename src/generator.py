import kenlm
import os
from pyvi import ViTokenizer

class NgramGenerator:
    def __init__(self, model_path, train_data_path, n_gram_order=5):
        print("Đang load mô hình KenLM...")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
        
        self.model = kenlm.Model(model_path)
        self.n_order = n_gram_order

        # Xây dựng bản đồ từ vựng (Next-word candidates)
        print("Đang xây dựng danh sách từ vựng (Candidates Map)...")
        self.vocab_map = {} 

        with open(train_data_path, "r", encoding="utf-8") as f:
            # --- Đọc từng dòng (từng bài) để xử lý riêng biệt ---
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Tách từ trong dòng hiện tại
                words = line.split()
                
                # QUAN TRỌNG: Thêm token kết thúc câu '</s>' vào cuối mỗi bài
                words.append("</s>") 

                # Lưu candidates cho dòng này
                for i in range(len(words) - 1):
                    for k in range(1, self.n_order): 
                        if i + k < len(words):
                            context_tuple = tuple(words[i : i+k]) 
                            next_word = words[i+k]

                            if context_tuple not in self.vocab_map:
                                self.vocab_map[context_tuple] = []
                            
                            # Thêm ứng viên nếu chưa có
                            if next_word not in self.vocab_map[context_tuple]:
                                self.vocab_map[context_tuple].append(next_word)

    def generate(self, seed_text, max_length=100):
        # Tách từ cho seed text
        current_text = ViTokenizer.tokenize(seed_text).lower()
        words = current_text.split()
        
        # print(f"Seed tokens: {words}") # Uncomment để debug

        for _ in range(max_length):
            candidates = None
            
            # --- CHIẾN THUẬT BACKOFF ---
            search_len = min(len(words), self.n_order - 1)
            
            for k in range(search_len, 0, -1):
                context_tuple = tuple(words[-k:]) 
                candidates = self.vocab_map.get(context_tuple)
                if candidates:
                    break
            
            if not candidates:
                # print(" -> Ngừng: Không tìm thấy từ tiếp theo.")
                break

            # --- DÙNG KENLM CHẤM ĐIỂM ---
            best_word = ""
            best_score = -9999
            
            for word in candidates:
                # Tạo câu giả định
                sentence = " ".join(words + [word])
                score = self.model.score(sentence)
                
                if score > best_score:
                    best_score = score
                    best_word = word
            
            # --- KIỂM TRA ĐIỀU KIỆN DỪNG ---
            if best_word == "</s>":
                # print(" -> Gặp tín hiệu kết thúc bài.")
                break 

            # Thêm từ tốt nhất vào danh sách
            words.append(best_word)

        # Ghép lại và xử lý hiển thị
        final_text = " ".join(words).replace("_", " ")
        return final_text
