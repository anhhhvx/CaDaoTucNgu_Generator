import os
from src.generator import NgramGenerator

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.bin')
    DATA_PATH = os.path.join(BASE_DIR, 'models', 'train_data_seg.txt')

    print("=== MÔ HÌNH SINH CA DAO TỤC NGỮ (KENLM) ===")
    
    try:
        # Khởi tạo Generator
        gen = NgramGenerator(MODEL_PATH, DATA_PATH, n_gram_order=5)
        print("\nModel đã sẵn sàng! Gõ 'exit' để thoát.")
        
        while True:
            seed = input("\nNhập câu gợi ý (VD: công cha): ")
            if seed.lower() in ['exit', 'quit']: break
            
            # Sinh văn bản
            result = gen.generate(seed)
            print(f"Kết quả: {result}")
            
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
        print("Bạn đã chạy 'python train.py' để tạo model chưa?")

if __name__ == "__main__":
    main()
