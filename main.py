import os
from src.generator import NgramGenerator

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.bin')
    DATA_PATH = os.path.join(BASE_DIR, 'models', 'train_data_seg.txt')

    print("=== MÔ HÌNH SINH CA DAO ĐA DẠNG (TOP-K) ===")
    
    try:
        gen = NgramGenerator(MODEL_PATH, DATA_PATH, n_gram_order=5)
        print("\nSẵn sàng! Gõ 'exit' để thoát.")
        
        while True:
            seed = input("\nNhập câu gợi ý (VD: công cha): ")
            if seed.lower() in ['exit', 'quit']: break
            
            # Hỏi muốn sinh bao nhiêu câu (Mặc định 5 câu)
            print(f"Đang suy nghĩ để tìm ra các biến thể khác nhau...")
            
            # Gọi hàm sinh hàng loạt (Batch)
            # top_k=5 nghĩa là nó sẽ cân nhắc 5 từ tốt nhất để ghép câu
            results = gen.generate_batch(seed, num_sentences=5, top_k=5)
            
            print(f"\n--- Tìm thấy {len(results)} kết quả ---")
            for i, sent in enumerate(results, 1):
                print(f"{i}. {sent}")
            
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
