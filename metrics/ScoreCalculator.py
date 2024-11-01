from common_imports import *
from metrics.CiderScorer import CiderScorer

def normalize_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text

def preprocess_sentence(sentence: str, tokenizer=None):
    sentence = sentence.lower()
    sentence = unicodedata.normalize('NFC', sentence)
    sentence = re.sub(r"[“”]", "\"", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    # tokenize the sentence
    if tokenizer is None:
        tokenizer = lambda s: s
    sentence = tokenizer(sentence)
    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()

    return tokens

class ScoreCalculator:
    def __init__(self):
        self.f1_caculate=F1()
        self.em_caculate=Exact_Match()
        self.Wup_caculate=Wup()
    #F1 score character level
    def f1_char(self,labels: List[str], preds: List[str]) -> float:
        scores=[]
        for i in range(len(labels)):
            scores.append(self.f1_caculate.compute_score(str(preprocess_sentence(normalize_text(labels[i]))).split(),str(preprocess_sentence(normalize_text(preds[i]))).split()))
        return np.mean(scores)

    #F1 score token level
    def f1_token(self,labels: List[str], preds: List[str]) -> float:
        scores=[]
        for i in range(len(labels)):
            scores.append(self.f1_caculate.compute_score(str(preprocess_sentence(normalize_text(labels[i]))).split(),str(preprocess_sentence(normalize_text(preds[i]))).split()))
        return np.mean(scores)
    #Excat match score
    def em(self,labels: List[str], preds: List[str]) -> float:
        scores=[]
        for i in range(len(labels)):
            scores.append(self.em_caculate.compute_score(str(preprocess_sentence(normalize_text(labels[i]))).split(),str(preprocess_sentence(normalize_text(preds[i]))).split()))
        return np.mean(scores)
    #Wup score
    def wup(self,labels: List[str], preds: List[str]) -> float:
        scores=[]
        for i in range(len(labels)):
            scores.append(self.Wup_caculate.compute_score(str(preprocess_sentence(normalize_text(labels[i]))).split(),str(preprocess_sentence(normalize_text(preds[i]))).split()))
        return np.mean(scores)
    #Cider score
    def cider_score(self,labels: List[str], preds: List[str]) -> float:
        labels=[[preprocess_sentence(normalize_text(label))] for label in labels]
        preds=[[preprocess_sentence(normalize_text(pred))] for pred in preds ]
        cider_caculate= CiderScorer(labels, test=preds, n=4, sigma=6.)
        scores,_=cider_caculate.compute_score()
        return scores
    
class F1:
  def Precision(self,y_true,y_pred):
    if y_pred is None:
       return 0
    common = set(y_true) & set(y_pred)
    return len(common) / len(set(y_pred))

  def Recall(self,y_true,y_pred):
    common = set(y_true) & set(y_pred)
    return len(common) / len(set(y_true))

  def compute_score(self,y_true,y_pred):
    if len(y_pred) == 0 or len(y_true) == 0:
        return int(y_pred == y_true)

    precision = self.Precision(y_true, y_pred)
    recall = self.Recall(y_true, y_pred)

    if precision == 0 or recall == 0:
        return 0
    f1 = 2*precision*recall / (precision+recall)
    return f1

class Exact_Match:
    def compute_score(self, y_true, y_pred):
        if y_true==y_pred:
            return 1
        else:
            return 0
        
class Wup:
    def get_semantic_field(self,a):
        weight = 1.0
        semantic_field = wordnet.synsets(str(a), pos=wordnet.NOUN)
        return (semantic_field,weight)

    def get_stem_word(self,a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)
    def compute_score(self, a: str, b: str, similarity_threshold: float = 0.9):
        """
        Returns Wu-Palmer similarity score.
        More specifically, it computes:
            max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
            where interp is a 'interpretation field'
        """
        global_weight=1.0

        (a,global_weight_a)=self.get_stem_word(a)
        (b,global_weight_b)=self.get_stem_word(b)
        global_weight = min(global_weight_a,global_weight_b)

        if a==b:
            # they are the same
            return 1.0*global_weight

        if a==[] or b==[]:
            return 0

        interp_a,weight_a = self.get_semantic_field(a)
        interp_b,weight_b = self.get_semantic_field(b)

        if interp_a == [] or interp_b == []:
            return 0

        # we take the most optimistic interpretation
        global_max=0.0
        for x in interp_a:
            for y in interp_b:
                local_score=x.wup_similarity(y)
                if local_score > global_max:
                    global_max=local_score

        # we need to use the semantic fields and therefore we downweight
        # unless the score is high which indicates both are synonyms
        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0

        final_score=global_max*weight_a*weight_b*interp_weight*global_weight
        return final_score