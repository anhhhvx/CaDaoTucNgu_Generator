import os
import subprocess
import sys
from src.preprocessing import normalize_and_tokenize

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(BASE_DIR, 'data', 'dataset_tho.txt')
SEG_DATA = os.path.join(BASE_DIR, 'models', 'train_data_seg.txt')
ARPA_FILE = os.path.join(BASE_DIR, 'models', 'model.arpa')
BIN_FILE = os.path.join(BASE_DIR, 'models', 'model.bin')

def main():
    print("BẮT ĐẦU HUẤN LUYỆN MODEL...")
    
    # BƯỚC 1: Kiểm tra dữ liệu thô
    if not os.path.exists(RAW_DATA):
        print(f"Không tìm thấy file dữ liệu gốc: {RAW_DATA}")
        print("Đang tạo file mẫu để demo...")
        sample_data = "Công cha như núi Thái Sơn\nNghĩa mẹ như nước trong nguồn chảy ra.\n\nBầu ơi thương lấy bí cùng\nTuy rằng khác giống nhưng chung một giàn."
        with open(RAW_DATA, "w", encoding="utf-8") as f:
            f.write(sample_data)
            
    # BƯỚC 2: Tiền xử lý (Chuẩn hóa + Tokenize)
    success = normalize_and_tokenize(RAW_DATA, SEG_DATA)
    if not success: return

    # BƯỚC 3: Gọi KenLM để train
    # Lưu ý về đường dẫn binary KenLM:
    # - Trên Colab: Thường nằm ở /content/kenlm/build/bin/
    # - Trên Local (Linux/Mac): Nếu đã cài make install thì chỉ cần gọi 'lmplz'
    
    # Ở đây ta ưu tiên đường dẫn Colab trước, nếu không thấy thì gọi lệnh hệ thống
    colab_bin_path = "/content/kenlm/build/bin/"
    
    lmplz_cmd = f"{colab_bin_path}lmplz" if os.path.exists(colab_bin_path) else "lmplz"
    build_binary_cmd = f"{colab_bin_path}build_binary" if os.path.exists(colab_bin_path) else "build_binary"
    
    try:
        print("\nĐang chạy lmplz (Tạo file ARPA)...")
        # Lệnh: lmplz -o 5 --text input --arpa output
        cmd_1 = f'{lmplz_cmd} -o 5 --text "{SEG_DATA}" --arpa "{ARPA_FILE}" --discount_fallback'
        subprocess.run(cmd_1, shell=True, check=True)
        
        print("Đang chạy build_binary (Nén sang Binary)...")
        # Lệnh: build_binary arpa_input binary_output
        cmd_2 = f'{build_binary_cmd} "{ARPA_FILE}" "{BIN_FILE}"'
        subprocess.run(cmd_2, shell=True, check=True)
        
        print(f"\nHUẤN LUYỆN THÀNH CÔNG! Model lưu tại: {BIN_FILE}")
        
    except subprocess.CalledProcessError:
        print("\nLỗi khi gọi KenLM.")
        print("Gợi ý: Nếu chạy trên máy tính (Local), hãy chắc chắn bạn đã cài đặt KenLM (C++) và thêm vào PATH.")
        print("Nếu chạy trên Colab, hãy chắc chắn bạn đã chạy cell biên dịch KenLM trước đó.")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

if __name__ == "__main__":
    main()
