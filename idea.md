# Ý tưởng cải tiến cho VLSP 2025 MLQA-TSR

## Tổng quan bài toán
VLSP 2025 MLQA-TSR là một shared task về Question Answering đa phương thức cho Traffic Sign Recognition, bao gồm:
- **Subtask 1**: Multimodal Retrieval - tìm kiếm điều khoản pháp luật liên quan
- **Subtask 2**: Question Answering - trả lời câu hỏi trắc nghiệm dựa trên context

## Approach hiện tại của team
1. Sử dụng VLM để mô tả chi tiết biển báo giao thông
2. Đưa mô tả vào LLM để trích xuất keywords
3. Sử dụng keywords để retrieve k context liên quan từ corpus pháp luật

## Các điểm cải tiến đề xuất

### 1. Cải tiến Architecture tổng thể

#### 1.1 Multi-stage Pipeline với Feedback Loop
```
Input (Image + Question) 
    ↓
Stage 1: Enhanced Visual Understanding
    ↓
Stage 2: Structured Information Extraction  
    ↓
Stage 3: Multi-modal Retrieval
    ↓
Stage 4: Context-aware QA
    ↓
Output + Confidence Score
    ↓
Feedback Loop (nếu confidence thấp)
```

#### 1.2 Fusion Strategy
- **Early Fusion**: Kết hợp visual features và text features từ đầu
- **Late Fusion**: Xử lý riêng rồi combine ở output layer
- **Cross-modal Attention**: Cho phép text và vision attend lẫn nhau

### 2. Cải tiến Visual Understanding

#### 2.1 Specialized Vision Encoder cho Traffic Signs
- **Pre-training**: Fine-tune vision encoder trên traffic sign dataset
- **Multi-scale Feature Extraction**: Trích xuất features ở nhiều scales
- **OCR Integration**: Tích hợp OCR để đọc text trên biển báo

#### 2.2 Structured Visual Description
Thay vì mô tả tự do, structure hóa thông tin:
```json
{
    "sign_type": "prohibition/warning/mandatory",
    "shape": "circle/triangle/rectangle", 
    "colors": ["red", "white", "blue"],
    "symbols": ["car", "truck", "no_entry"],
    "text_content": "6:00-22:00",
    "size_info": "5T",
    "context": "urban_road/highway"
}
```

### 3. Cải tiến Information Extraction & Retrieval

#### 3.1 Hierarchical Keyword Extraction
```
Level 1: Traffic Sign Categories (cấm, chỉ dẫn, cảnh báo)
Level 2: Vehicle Types (xe con, xe tải, xe khách)  
Level 3: Time/Weight/Size Constraints
Level 4: Specific Regulations (điều, khoản)
```

#### 3.2 Enhanced Retrieval Strategy
- **Dense Retrieval**: Sử dụng embedding models (VietAI, PhoBERT-based)
- **Hybrid Retrieval**: Kết hợp BM25 và dense retrieval
- **Re-ranking**: Sử dụng cross-encoder để re-rank candidates
- **Query Expansion**: Mở rộng query dựa trên visual information

#### 3.3 Multi-modal Embedding
- Tạo shared embedding space cho cả visual và textual information
- Contrastive learning để align traffic sign images với legal text

### 4. Cải tiến Training Strategy

#### 4.1 Multi-task Learning
```python
# Pseudo code
total_loss = α * retrieval_loss + β * qa_loss + γ * auxiliary_losses

auxiliary_losses = {
    "sign_classification_loss": classify traffic sign types,
    "text_detection_loss": detect text in images,
    "regulation_entity_loss": extract legal entities
}
```

#### 4.2 Data Augmentation
- **Image Augmentation**: Rotation, lighting changes, weather effects
- **Question Paraphrasing**: Tạo các cách hỏi khác nhau cho cùng một intent
- **Legal Text Augmentation**: Synthetic negative samples

#### 4.3 Progressive Training
1. **Stage 1**: Pre-train VLM trên general Vietnamese VQA data
2. **Stage 2**: Fine-tune trên traffic sign description task
3. **Stage 3**: Joint training cho retrieval và QA
4. **Stage 4**: End-to-end fine-tuning với hard negatives

### 5. Specific Vietnamese Optimizations

#### 5.1 Vietnamese Language Model Selection
- **Backbone options**: VietAI/envit5-translation, UIT-NLDB/viT5-base
- **Embedding models**: VietAI/vietnamese-bi-encoder
- **Consider**: PhoGPT, Vintern-1B (nếu được phép)

#### 5.2 Vietnamese Legal Text Processing
- **Named Entity Recognition**: Trích xuất số điều, khoản, thông tư
- **Legal Term Normalization**: Chuẩn hóa thuật ngữ pháp lý
- **Regex Patterns**: Cho các format chuẩn như "Điều X.Y", "Thông tư Z"

### 6. Advanced Techniques

#### 6.1 Chain-of-Thought cho Legal Reasoning
```
1. Identify traffic sign elements
2. Map to relevant legal categories  
3. Search for specific regulations
4. Apply context to answer question
```

#### 6.2 Retrieval-Augmented Generation (RAG) Enhancement
- **Contextual Re-ranking**: Sử dụng question context để re-rank
- **Multi-hop Reasoning**: Kết hợp nhiều điều luật liên quan
- **Confidence Calibration**: Đánh giá độ tin cậy của retrieval

#### 6.3 Knowledge Graph Integration
Xây dựng KG cho traffic regulations:
```
Traffic_Sign → has_meaning → Regulation
Regulation → applies_to → Vehicle_Type  
Vehicle_Type → restricted_by → Time_Constraint
```

### 7. Training Data Strategy

#### 7.1 Data Collection & Annotation
- **Synthetic Data**: Tạo synthetic traffic scenarios
- **Crowd-sourcing**: Thu thập annotations chất lượng cao
- **Expert Validation**: Legal experts validate annotations

#### 7.2 Hard Negative Mining
- Mine hard negatives từ similar traffic signs
- Cross-regulation confusions (regulations that look similar)
- Visual ambiguity cases

### 8. Model Architecture Recommendations

#### 8.1 Baseline Architecture
```python
class TrafficSignQA(nn.Module):
    def __init__(self):
        self.vision_encoder = EVA_CLIP_Vision()
        self.text_encoder = PhoBERT()
        self.cross_modal_fusion = CrossAttentionBlock()
        self.retrieval_head = RetrievalHead()
        self.qa_head = QAHead()
        
    def forward(self, image, question, legal_docs):
        # Multi-modal encoding và processing
        pass
```

#### 8.2 Advanced Architecture với Memory
- **External Memory**: Store traffic sign prototypes
- **Episodic Memory**: Remember previous similar cases
- **Working Memory**: Maintain context during reasoning

### 9. Evaluation & Debugging

#### 9.1 Component-wise Evaluation
- Visual understanding accuracy
- Keyword extraction precision/recall  
- Retrieval performance (MRR, NDCG)
- End-to-end QA accuracy

#### 9.2 Error Analysis Framework
- **Visual errors**: Misclassified signs, missed text
- **Retrieval errors**: Wrong legal documents retrieved
- **Reasoning errors**: Correct info but wrong conclusion

### 10. Implementation Timeline

#### Phase 1 (2 weeks): Data Preparation & Baseline
- Data collection and preprocessing
- Implement baseline VLM + LLM pipeline
- Basic retrieval system

#### Phase 2 (3 weeks): Core Improvements  
- Enhanced visual understanding
- Improved retrieval mechanism
- Multi-task training setup

#### Phase 3 (2 weeks): Advanced Features
- Knowledge graph integration
- Chain-of-thought reasoning
- Model ensemble

#### Phase 4 (1 week): Final Optimization
- Hyperparameter tuning
- Model selection and ensemble
- Submission preparation

## Key Success Factors

1. **High-quality Vietnamese Legal Corpus**: Đảm bảo coverage đầy đủ
2. **Robust Visual Understanding**: Xử lý được variations in lighting, angle, weather
3. **Effective Multi-modal Fusion**: Tận dụng được cả visual và textual signals
4. **Strong Vietnamese Language Model**: Hiểu được legal terminology
5. **Efficient Retrieval System**: Balance between precision và recall

## Rủi ro và Mitigation

### Rủi ro
- Dataset size limitations
- Vietnamese legal text complexity
- Visual quality variations
- Computational constraints

### Mitigation
- Data augmentation strategies
- Transfer learning from English models
- Robust preprocessing pipeline
- Model efficiency optimizations

---

*Note: Approach này có thể được điều chỉnh dựa trên dataset thực tế và computational resources available.* 