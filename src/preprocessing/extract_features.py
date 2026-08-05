import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

def test_siglip():
    # 1. Định nghĩa model SigLIP (Sử dụng bản base-patch16-224 để tải nhanh khi test)
    model_name = "google/siglip-base-patch16-224"
    print(f"Đang tải mô hình SigLIP: {model_name}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    print(f"Đã load model thành công trên thiết bị: {device}")

    # 2. Tạo dữ liệu giả lập để test
    # Tạo một ảnh ngẫu nhiên kích thước 224x224 để test (trong thực tế sẽ load ảnh từ đường dẫn)
    print("Tạo ảnh test ngẫu nhiên...")
    test_image = Image.new('RGB', (224, 224), color = (73, 109, 137))
    
    # Định nghĩa câu mô tả để so khớp
    test_texts = [
        "a blue background image", 
        "a red car on the street", 
        "a speaker in a conference"
    ]

    # 3. Tiền xử lý (Tokenize text và Process image)
    inputs = processor(
        text=test_texts, 
        images=test_image, 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)

    # 4. Trích xuất Embeddings và tính toán tương đồng
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Lấy đặc trưng dạng normalized (đã L2-normalized sẵn để tính Cosine Similarity trực tiếp)
        image_embeds = outputs.image_embeds # Shape: [1, feature_dim]
        text_embeds = outputs.text_embeds   # Shape: [num_texts, feature_dim]
        
        # Tính tương đồng Cosine bằng phép nhân ma trận (Dot Product)
        # Vì SigLIP embeds đã được normalized, dot product chính là Cosine Similarity
        similarity = torch.matmul(image_embeds, text_embeds.T) # Shape: [1, num_texts]
        
    print("\n--- KẾT QUẢ TEST SIGLIP ---")
    print(f"Kích thước Vector ảnh: {image_embeds.shape}")
    print(f"Kích thước Vector text: {text_embeds.shape}")
    print("\nĐiểm số tương đồng giữa ảnh test (màu xanh lam) và các câu mô tả:")
    for text, score in zip(test_texts, similarity[0].tolist()):
        print(f" - '{text}': {score:.4f}")

if __name__ == "__main__":
    test_siglip()

