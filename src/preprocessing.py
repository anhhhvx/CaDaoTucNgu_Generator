import re
import os
from pyvi import ViTokenizer

def normalize_and_tokenize(input_path, output_path):
    print(f"Đang tiền xử lý dữ liệu từ: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file {input_path}")
        return False

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            
        # 1. Tách các bài thơ bằng dòng trống (2 lần xuống dòng trở lên)
        poems = re.split(r'\n\s*\n', raw_content.strip())
        normalized_lines = []
        
        for poem in poems:
            verses = poem.strip().split('\n')
            # Làm sạch, chuyển chữ thường, giữ dấu câu
            clean_verses = [v.strip().lower() for v in verses if v.strip()]
            
            if clean_verses:
                # Nối các câu thơ bằng dấu chấm
                merged_line = ". ".join(clean_verses)
                if not merged_line.endswith('.'):
                    merged_line += "."
                # Xử lý lỗi 2 dấu chấm
                merged_line = merged_line.replace("..", ".")
                
                # 2. Tokenize (Tách từ) ngay lập tức bằng PyVi
                # Ví dụ: "công cha" -> "công cha" (PyVi giữ nguyên vì là từ đơn)
                # "núi thái sơn" -> "núi thái_sơn"
                tokenized_line = ViTokenizer.tokenize(merged_line)
                
                normalized_lines.append(tokenized_line)
                
        # Lưu kết quả ra file để KenLM huấn luyện
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(normalized_lines))
            
        print(f"Đã xử lý xong {len(normalized_lines)} bài. Lưu tại: {output_path}")
        return True
        
    except Exception as e:
        print(f"Có lỗi khi xử lý dữ liệu: {e}")
        return False
