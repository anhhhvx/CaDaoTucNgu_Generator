import kenlm
import os
import random
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

    def generate(self, seed_text, max_length=100, top_k=3):
        """
        Sinh ra 1 câu duy nhất nhưng có tính ngẫu nhiên nhờ top_k.
        top_k=1: Luôn chọn từ tốt nhất (giống cũ).
        top_k=3: Chọn ngẫu nhiên trong 3 từ tốt nhất (tạo sự đa dạng).
        """
        # Tách từ cho seed text
        current_text = ViTokenizer.tokenize(seed_text).lower()
        words = current_text.split()
        
        # print(f"Seed tokens: {words}") # Uncomment để debug

        for _ in range(max_length):
            candidates = None
            
            # --- CHIẾN THUẬT BACKOFF ---
            search_len = min(len(words), self.n_order - 1)
            # Backoff tìm candidates
            for k in range(search_len, 0, -1):
                context_tuple = tuple(words[-k:]) 
                candidates = self.vocab_map.get(context_tuple)
                if candidates:
                    break
            
            if not candidates:
                # print(" -> Ngừng: Không tìm thấy từ tiếp theo.")
                break

            # --- SỬA ĐỔI CHÍNH Ở ĐÂY: Thêm tham số top_k và temperature ---
    def generate_one(self, seed_text, max_length=100, top_k=3):
        """
        Sinh ra 1 câu duy nhất nhưng có tính ngẫu nhiên nhờ top_k.
        top_k=1: Luôn chọn từ tốt nhất (giống cũ).
        top_k=3: Chọn ngẫu nhiên trong 3 từ tốt nhất (tạo sự đa dạng).
        """
        current_text = ViTokenizer.tokenize(seed_text).lower()
        words = current_text.split()
        
        for _ in range(max_length):
            candidates = None
            search_len = min(len(words), self.n_order - 1)
            
            # Backoff tìm candidates
            for k in range(search_len, 0, -1):
                context_tuple = tuple(words[-k:]) 
                candidates = self.vocab_map.get(context_tuple)
                if candidates: break
            
            if not candidates: break

            # --- LOGIC MỚI: CHẤM ĐIỂM VÀ LẤY TOP K ---
            scored_candidates = []
            unique_candidates = list(set(candidates)) # Loại bỏ từ trùng lặp

            for word in unique_candidates:
                sentence = " ".join(words + [word])
                score = self.model.score(sentence)
                scored_candidates.append((score, word))
            
            # Sắp xếp từ điểm cao xuống thấp
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # Lấy Top K ứng viên tốt nhất (Ví dụ lấy 3 từ điểm cao nhất)
            # Nếu danh sách ít hơn k thì lấy hết
            actual_k = min(len(scored_candidates), top_k)
            top_choices = scored_candidates[:actual_k]

            # Chọn ngẫu nhiên 1 từ trong nhóm Top K này
            best_word = random.choice(top_choices)[1]
            
            if best_word == "</s>": break 
            words.append(best_word)

        return " ".join(words).replace("_", " ")

    # --- HÀM MỚI: SINH NHIỀU CÂU CÙNG LÚC ---
    def generate_batch(self, seed_text, num_sentences=5, top_k=3):
        """
        Sinh ra danh sách nhiều câu khác nhau từ cùng 1 gợi ý.
        """
        results = set() # Dùng set để tự động loại bỏ câu trùng nhau
        attempts = 0
        max_attempts = num_sentences * 3 # Tránh vòng lặp vô tận nếu model quá ít dữ liệu
        
        while len(results) < num_sentences and attempts < max_attempts:
            sentence = self.generate_one(seed_text, top_k=top_k)
            results.add(sentence)
            attempts += 1
            
        return list(results)
