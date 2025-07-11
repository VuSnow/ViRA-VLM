# VLSP 2025 MLQA-TSR - Implementation Checklist

## Phase 1: Data Preparation & Infrastructure (2 weeks)

### ✅ Task 1: Dataset Analysis & Preprocessing

**Công nghệ sử dụng:**
- `pandas`, `PIL/OpenCV` cho image processing
- `transformers` tokenizers cho text preprocessing  
- `matplotlib/seaborn` cho data visualization

**Mục đích:** 
Hiểu rõ structure và characteristics của dataset, chuẩn bị data pipeline

**Input:**
- Raw VLSP 2025 dataset (images + questions + legal documents)
- Competition guidelines và evaluation metrics

**Output:**
- Dataset statistics report (image resolutions, question types, answer distributions)
- Cleaned và normalized dataset
- Data split strategy (train/val/test)

**Chi tiết implementation:**
```python
# Dataset statistics
analyze_image_properties()  # Resolution, aspect ratio, file sizes
analyze_question_patterns()  # Question types, length distribution
analyze_legal_corpus()     # Document structure, entity patterns
```

---

### ✅ Task 2: Vietnamese Legal Corpus Processing

**Công nghệ sử dụng:**
- `spaCy` với Vietnamese model hoặc `VnCoreNLP`
- `regex` cho pattern extraction
- `elasticsearch` hoặc `faiss` cho indexing

**Mục đích:**
Xây dựng searchable legal knowledge base với proper indexing

**Input:**
- Vietnamese traffic law documents
- Regulation texts (Thông tư 54/2019/TT-BGTVT, etc.)

**Output:**
- Structured legal database với metadata
- Entity extraction patterns cho legal references
- Search index cho retrieval system

**Chi tiết implementation:**
```python
# Legal text processing
extract_legal_entities()    # "Điều 26.1", "P.106(a,b)"
normalize_legal_terms()     # Standardize terminology
create_legal_hierarchy()    # Article → Section → Paragraph
build_search_index()        # For fast retrieval
```

---

### ✅ Task 3: Baseline VLM Setup

**Công nghệ sử dụng:**
- `transformers` library
- Pre-trained VLM models: `blip2-opt-2.7b`, `instructblip-vicuna-7b`
- `torch` với CUDA support

**Mục đích:**
Establish baseline performance cho visual understanding task

**Input:**
- Traffic sign images từ dataset
- Template prompts cho description generation

**Output:**
- Baseline VLM pipeline
- Performance metrics trên validation set
- Sample descriptions để analyze quality

**Chi tiết implementation:**
```python
# VLM baseline
class BaselineVLM:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    def describe_traffic_sign(self, image, prompt):
        # Generate description with specific prompt
        pass
```

---

## Phase 2: Core Model Development (3 weeks)

### ✅ Task 4: Enhanced Vision Encoder

**Công nghệ sử dụng:**
- `EVA-CLIP` hoặc `CLIP-ViT` pre-trained models
- `timm` library cho additional vision models
- `albumentations` cho augmentation

**Mục đích:**
Improve visual understanding specifically cho traffic signs

**Input:**
- Traffic sign images với various conditions (lighting, angles, weather)
- Optional: Additional traffic sign datasets cho pre-training

**Output:**
- Fine-tuned vision encoder
- Improved visual feature representations
- Performance comparison với baseline

**Chi tiết implementation:**
```python
# Enhanced vision processing
class TrafficSignVisionEncoder:
    def __init__(self):
        self.clip_model = EVACLIPModel.from_pretrained("EVA-CLIP")
        self.ocr_model = EasyOCR(['vi', 'en'])  # For text in signs
        
    def extract_visual_features(self, image):
        # Multi-scale feature extraction
        clip_features = self.clip_model.encode_image(image)
        ocr_text = self.ocr_model.readtext(image)
        return combine_features(clip_features, ocr_text)
```

---

### ✅ Task 5: Structured Information Extraction

**Công nghệ sử dụng:**
- Custom `torch.nn.Module` cho structured output
- `pydantic` cho data validation
- `json-schema` cho output format

**Mục đích:**
Replace free-form description với structured information extraction

**Input:**
- Raw traffic sign images
- Visual features từ enhanced vision encoder

**Output:**
- Structured JSON với traffic sign attributes
- Improved consistency trong information extraction

**Chi tiết implementation:**
```python
# Structured extraction
class StructuredExtractor(nn.Module):
    def __init__(self, vision_dim, hidden_dim):
        self.sign_type_classifier = nn.Linear(vision_dim, 3)  # prohibition/warning/mandatory
        self.shape_classifier = nn.Linear(vision_dim, 4)      # circle/triangle/rectangle/diamond
        self.color_detector = nn.Linear(vision_dim, 10)       # multi-label for colors
        self.text_extractor = TextExtractionHead(vision_dim)
        
    def forward(self, visual_features):
        return {
            "sign_type": self.sign_type_classifier(visual_features),
            "shape": self.shape_classifier(visual_features), 
            "colors": self.color_detector(visual_features),
            "text_content": self.text_extractor(visual_features)
        }
```

---

### ✅ Task 6: Hierarchical Keyword Extraction

**Công nghệ sử dụng:**
- `PhoBERT` hoặc `VietAI/vietnamese-bi-encoder`
- `transformers` cho sequence labeling
- Custom NER models cho legal entities

**Mục đích:**
Extract hierarchical keywords cho improved retrieval performance

**Input:**
- Structured visual information từ Task 5
- Input questions từ users

**Output:**
- Multi-level keyword hierarchy
- Weighted importance scores cho mỗi keyword
- Improved retrieval queries

**Chi tiết implementation:**
```python
# Hierarchical keyword extraction
class HierarchicalKeywordExtractor:
    def __init__(self):
        self.phobert = PhoBERT.from_pretrained("vinai/phobert-base")
        self.legal_ner = load_legal_ner_model()
        
    def extract_keywords(self, structured_info, question):
        # Level 1: Traffic sign categories
        category_keywords = self.extract_categories(structured_info)
        
        # Level 2: Vehicle types mentioned
        vehicle_keywords = self.extract_vehicles(question, structured_info)
        
        # Level 3: Constraints (time, weight, size)
        constraint_keywords = self.extract_constraints(structured_info)
        
        # Level 4: Legal references
        legal_keywords = self.legal_ner.extract(question)
        
        return {
            "level_1": category_keywords,
            "level_2": vehicle_keywords, 
            "level_3": constraint_keywords,
            "level_4": legal_keywords,
            "weights": self.calculate_weights()
        }
```

---

### ✅ Task 7: Multi-modal Retrieval System

**Công nghệ sử dụng:**
- `sentence-transformers` với Vietnamese models
- `faiss` cho efficient similarity search
- `rank_bm25` cho keyword-based retrieval
- `transformers` cross-encoder cho re-ranking

**Mục đích:**
Build robust retrieval system combining visual và textual information

**Input:**
- Hierarchical keywords từ Task 6
- Legal document corpus từ Task 2
- Original question và visual features

**Output:**
- Top-k relevant legal documents
- Confidence scores cho mỗi retrieved document
- Explanation của retrieval decisions

**Chi tiết implementation:**
```python
# Multi-modal retrieval
class MultiModalRetriever:
    def __init__(self):
        self.dense_retriever = SentenceTransformer('VietAI/vietnamese-bi-encoder')
        self.sparse_retriever = BM25Okapi()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def retrieve(self, keywords, question, visual_features, k=10):
        # Dense retrieval
        dense_query = self.create_dense_query(keywords, question)
        dense_candidates = self.dense_retriever.search(dense_query, k*2)
        
        # Sparse retrieval  
        sparse_query = self.create_sparse_query(keywords)
        sparse_candidates = self.sparse_retriever.search(sparse_query, k*2)
        
        # Combine và re-rank
        combined_candidates = self.combine_results(dense_candidates, sparse_candidates)
        reranked_results = self.reranker.rank(question, combined_candidates)
        
        return reranked_results[:k]
```

---

## Phase 3: Advanced Features (2 weeks)

### ✅ Task 8: Multi-task Learning Framework

**Công nghệ sử dụng:**
- `pytorch-lightning` cho training framework
- Custom loss functions cho multi-task objectives
- `wandb` cho experiment tracking

**Mục đích:**
Improve model performance through joint training của multiple related tasks

**Input:**
- Training data cho retrieval task
- Training data cho QA task  
- Auxiliary task data (sign classification, etc.)

**Output:**
- Joint model với improved performance
- Ablation study results
- Best hyperparameters cho multi-task setup

**Chi tiết implementation:**
```python
# Multi-task learning
class MultiTaskModel(pl.LightningModule):
    def __init__(self, config):
        self.vision_encoder = EnhancedVisionEncoder()
        self.text_encoder = PhoBERTEncoder()
        self.fusion_layer = CrossModalFusion()
        self.retrieval_head = RetrievalHead()
        self.qa_head = QAHead()
        self.auxiliary_heads = {
            'sign_classification': SignClassificationHead(),
            'text_detection': TextDetectionHead()
        }
        
    def training_step(self, batch, batch_idx):
        # Joint training với weighted losses
        retrieval_loss = self.compute_retrieval_loss(batch)
        qa_loss = self.compute_qa_loss(batch) 
        aux_losses = self.compute_auxiliary_losses(batch)
        
        total_loss = (self.config.alpha * retrieval_loss + 
                     self.config.beta * qa_loss + 
                     self.config.gamma * sum(aux_losses.values()))
        
        return total_loss
```

---

### ✅ Task 9: Chain-of-Thought Reasoning

**Công nghệ sử dụng:**
- Custom prompt engineering cho CoT
- `transformers` generation với intermediate steps
- Reasoning path validation logic

**Mục đích:**
Enable step-by-step legal reasoning cho better interpretability và accuracy

**Input:**
- Retrieved legal documents từ Task 7
- Original question
- Visual understanding results

**Output:**
- Step-by-step reasoning process
- Final answer với explanation
- Confidence score cho reasoning chain

**Chi tiết implementation:**
```python
# Chain-of-thought reasoning
class LegalCoTReasoner:
    def __init__(self):
        self.reasoning_templates = {
            'sign_identification': "Bước 1: Nhận diện biển báo...",
            'legal_mapping': "Bước 2: Tìm điều luật liên quan...", 
            'context_application': "Bước 3: Áp dụng ngữ cảnh...",
            'answer_generation': "Bước 4: Đưa ra kết luận..."
        }
        
    def reason(self, question, visual_info, legal_docs):
        reasoning_chain = []
        
        # Step 1: Identify traffic sign elements
        step1 = self.identify_sign_elements(visual_info)
        reasoning_chain.append(step1)
        
        # Step 2: Map to legal categories
        step2 = self.map_to_legal_categories(step1, legal_docs)
        reasoning_chain.append(step2)
        
        # Step 3: Apply contextual constraints
        step3 = self.apply_context(step2, question)
        reasoning_chain.append(step3)
        
        # Step 4: Generate final answer
        final_answer = self.generate_answer(step3, question)
        
        return {
            'reasoning_chain': reasoning_chain,
            'final_answer': final_answer,
            'confidence': self.calculate_confidence(reasoning_chain)
        }
```

---

### ✅ Task 10: Knowledge Graph Integration

**Công nghệ sử dụng:**
- `neo4j` hoặc `networkx` cho graph storage
- `spaCy` relation extraction
- Graph neural networks (`torch-geometric`)

**Mục đích:**
Enhance reasoning với structured knowledge về traffic regulations

**Input:**
- Processed legal documents
- Traffic sign taxonomy
- Relationship patterns từ legal text

**Output:**
- Traffic regulation knowledge graph
- Graph-enhanced reasoning capabilities
- Improved multi-hop reasoning

**Chi tiết implementation:**
```python
# Knowledge graph construction
class TrafficRegulationKG:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_types = ['TrafficSign', 'Regulation', 'VehicleType', 'TimeConstraint', 'SizeConstraint']
        
    def build_graph(self, legal_docs, sign_taxonomy):
        # Extract entities và relationships
        entities = self.extract_entities(legal_docs)
        relations = self.extract_relations(legal_docs)
        
        # Build graph structure
        for entity in entities:
            self.graph.add_node(entity.id, type=entity.type, properties=entity.props)
            
        for relation in relations:
            self.graph.add_edge(relation.source, relation.target, type=relation.type)
            
    def query_graph(self, query_entities, max_hops=3):
        # Graph traversal cho multi-hop reasoning
        return self.traverse_graph(query_entities, max_hops)
```

---

## Phase 4: Final Optimization (1 week)

### ✅ Task 11: Model Ensemble & Selection

**Công nghệ sử dụng:**
- `sklearn.ensemble` techniques adapted for deep learning
- Custom voting mechanisms
- Performance analysis tools

**Mục đích:**
Combine multiple models cho best possible performance

**Input:**
- Multiple trained models từ different approaches
- Validation performance data
- Computational efficiency metrics

**Output:**
- Optimal ensemble configuration
- Final model selection
- Performance benchmarks

**Chi tiết implementation:**
```python
# Model ensemble
class ModelEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0/len(models)] * len(models)
        
    def predict(self, inputs):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(inputs)
            predictions.append(pred * weight)
            
        # Weighted voting cho retrieval
        retrieval_scores = self.combine_retrieval_scores(predictions)
        
        # Ensemble cho QA
        qa_predictions = self.ensemble_qa_predictions(predictions)
        
        return {
            'retrieval': retrieval_scores,
            'qa': qa_predictions
        }
```

---

### ✅ Task 12: Comprehensive Evaluation

**Công nghệ sử dụng:**
- Custom evaluation metrics cho F1 (retrieval) và Accuracy (QA)
- `matplotlib/seaborn` cho visualization
- Statistical significance testing

**Mục đích:**
Thorough evaluation và error analysis cho final submission

**Input:**
- Final model predictions
- Ground truth labels
- Component-wise performance data

**Output:**
- Comprehensive evaluation report
- Error analysis với actionable insights
- Competition submission files

**Chi tiết implementation:**
```python
# Comprehensive evaluation
class ComprehensiveEvaluator:
    def __init__(self):
        self.metrics = {
            'retrieval': ['precision', 'recall', 'f1', 'mrr', 'ndcg'],
            'qa': ['accuracy', 'per_class_accuracy', 'confusion_matrix']
        }
        
    def evaluate(self, predictions, ground_truth):
        results = {}
        
        # Retrieval evaluation
        results['retrieval'] = self.evaluate_retrieval(
            predictions['retrieval'], 
            ground_truth['legal_refs']
        )
        
        # QA evaluation  
        results['qa'] = self.evaluate_qa(
            predictions['qa'],
            ground_truth['answers']
        )
        
        # Error analysis
        results['error_analysis'] = self.analyze_errors(predictions, ground_truth)
        
        return results
```

---

## Continuous Tasks

### 📊 Task 13: Experiment Tracking & Documentation

**Công nghệ sử dụng:**
- `wandb` hoặc `tensorboard` cho experiment tracking
- `jupyter notebooks` cho analysis
- `git` với proper branching strategy

**Mục đích:**
Maintain reproducibility và track progress throughout development

**Input:**
- Experiment configurations
- Training metrics
- Model performance data

**Output:**
- Comprehensive experiment logs
- Reproducible model checkpoints
- Technical documentation

---

### 🔧 Task 14: Code Quality & Testing

**Công nghệ sử dụng:**
- `pytest` cho unit testing
- `black` và `isort` cho code formatting
- `mypy` cho type checking

**Mục đích:**
Ensure code quality và maintainability

**Input:**
- All developed code modules
- Test cases và edge cases

**Output:**
- Clean, tested, documented codebase
- CI/CD pipeline setup
- Ready-to-submit code

---

## Success Metrics

### Retrieval Task (Subtask 1)
- **Target F1**: > 0.85
- **Precision**: > 0.80  
- **Recall**: > 0.90

### QA Task (Subtask 2)  
- **Target Accuracy**: > 0.90
- **Per-class performance**: Balanced across A/B/C/D options

### Technical Metrics
- **Inference time**: < 5 seconds per sample
- **Memory usage**: < 8GB GPU memory
- **Model size**: < 7B parameters (efficiency requirement)

---

## Key Dependencies & Setup

### Environment Setup
```bash
# Python dependencies
pip install torch torchvision transformers
pip install sentence-transformers faiss-cpu
pip install spacy vncorelp
pip install albumentations opencv-python
pip install wandb tensorboard
pip install pytest black isort mypy

# Vietnamese language model
python -m spacy download vi_core_news_lg
```

### Hardware Requirements
- **GPU**: NVIDIA RTX 3090/4090 (24GB VRAM) hoặc equivalent
- **RAM**: 32GB+ system memory
- **Storage**: 500GB+ SSD cho datasets và model checkpoints

---

**Note:** Checklist này sẽ được update based on intermediate results và feedback từ evaluation trên validation set.
