"""Evaluation metrics and COCO dataset utilities for TAM.

Implements Obj-IoU, Func-IoU, F1-IoU from the TAM paper (Eq. 8-10).
"""

import os
import json
import cv2
import numpy as np
import string
import unicodedata

import nltk
from nltk.stem import WordNetLemmatizer

from .config import FUNCTION_WORD_TAGS, NOUN_TAGS, COCO_CATEGORIES

# Ensure NLTK data is available
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

_lemmatizer = WordNetLemmatizer()


def get_word_type(word):
    """Classify word as 'function', 'noun', or 'others' using POS tagging."""
    tagged = nltk.pos_tag([word])
    pos = tagged[0][1]
    if pos in FUNCTION_WORD_TAGS:
        return 'function'
    elif pos in NOUN_TAGS:
        return 'noun'
    return 'others'


def single_words_match(word1, word2):
    """Check if two words match after lemmatization."""
    a = _lemmatizer.lemmatize(word1.lower().replace('-', ''))
    b = _lemmatizer.lemmatize(word2.lower().replace('-', ''))
    return a == b


def words_match(category_word, target_word):
    """Check if any word in category_word matches target_word."""
    for tk in category_word.split():
        if single_words_match(tk, target_word):
            return True
    return False


def is_english_punctuation(char):
    return char in string.punctuation


def is_chinese_char_or_punctuation(char):
    for ch in char:
        if 'CJK' in unicodedata.name(ch, ''):
            return True
    return False


def ids_to_word_groups(ids, processor):
    """Decode token IDs into grouped words with their token indices.

    Returns:
        (words, tokens_idx): list of word strings and list of token index lists.
    """
    txt = processor.batch_decode(ids)[0]
    tokens = processor.tokenizer.tokenize(txt)
    words, tokens_idx = [], []
    for i, tok in enumerate(tokens):
        word = processor.tokenizer.decode(processor.tokenizer.convert_tokens_to_ids(tok))
        if (i == 0 or is_english_punctuation(word) or
                is_chinese_char_or_punctuation(word) or
                word[0] == ' ' or tok[0] == '▁'):
            words.append(word.replace(' ', ''))
            tokens_idx.append([i])
        else:
            words[-1] += word.replace(' ', '')
            tokens_idx[-1].append(i)
    return words, tokens_idx


def evaluate(maps, tokens, processor, caption, mask_path, category):
    """Evaluate activation maps against ground-truth masks.

    Computes Obj-IoU (Eq. 8), Func-IoU (Eq. 9), and NLG metrics.

    Args:
        maps: List of activation map arrays, one per generated token.
        tokens: Generated token IDs (trimmed, answer only).
        processor: Model processor for decoding.
        caption: List of reference caption strings.
        mask_path: Path to ground-truth segmentation mask (grayscale PNG).
        category: Dict mapping category names to label IDs.

    Returns:
        [obj_iou_list, func_iou_list, rougel_list, meteor_list, precision_list, recall_list]
    """
    words, tokens_id = ids_to_word_groups(tokens, processor)
    if tokens_id[-1][-1] != (len(maps) - 1):
        return [[], [], [], [], [], []]

    # Classify words
    words_label = []
    for word in words:
        wtype = get_word_type(word)
        if wtype == 'noun':
            lb = -1
            for k, v in category.items():
                if words_match(k, word):
                    lb = v
            words_label.append(lb)
        elif wtype == 'function':
            words_label.append(-2)
        else:
            words_label.append(-3)

    # Load mask
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.zeros_like(maps[0])

    # Obj-IoU
    obj_iou, pre, rec, noun_fg_thresh = [], [], [], []
    for i in range(len(words)):
        if words_label[i] > 0:
            ious, pres, recs, thresh = [], [], [], []
            gt = (mask == words_label[i]).astype('uint8')
            for j in tokens_id[i]:
                m = cv2.resize(maps[j], (mask.shape[1], mask.shape[0]))
                t, pred = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if gt.sum() != 0:
                    tp = float((gt * pred > 0).sum())
                    ious.append(tp / ((gt + pred / 255) > 0).sum())
                    pres.append(tp / max((pred > 0).sum(), 1))
                    recs.append(tp / max((gt > 0).sum(), 1))
                thresh.append(t)

            noun_fg_thresh.append(max(thresh))
            if len(ious) > 0:
                m_iou = max(ious)
                obj_iou.append(m_iou)
                pre.append(pres[ious.index(m_iou)])
                rec.append(recs[ious.index(m_iou)])

            # Merge consecutive same-category words
            if len(obj_iou) > 1 and words_label[i] > 0 and i > 0 and words_label[i - 1] == words_label[i]:
                select_idx = -1 if obj_iou[-1] > obj_iou[-2] else -2
                obj_iou[-2] = obj_iou[select_idx]
                obj_iou = obj_iou[:-1]
                pre[-2] = pre[select_idx]
                pre = pre[:-1]
                rec[-2] = rec[select_idx]
                rec = rec[:-1]

        elif words_label[i] == -1:
            thresh = []
            for j in tokens_id[i]:
                m = cv2.resize(maps[j], (mask.shape[1], mask.shape[0]))
                t, _ = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh.append(t)
            noun_fg_thresh.append(max(thresh))

    # Func-IoU
    func_iou = []
    if len(noun_fg_thresh) > 0:
        fg_thresh = sum(noun_fg_thresh) / len(noun_fg_thresh)
        for i in range(len(words)):
            if words_label[i] == -2:
                neg_iou = []
                for j in tokens_id[i]:
                    neg_iou.append(float((maps[j] < fg_thresh).sum()) / maps[j].size)
                func_iou.append(sum(neg_iou) / len(neg_iou))

    # NLG metrics (optional, graceful failure)
    rougel, meteor = [], []
    try:
        from nltk.translate import meteor_score as ms
        from rouge import Rouge

        output_text = processor.batch_decode(tokens, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        ref = [str(c).lower().split() for c in caption]
        hypo = str(output_text[0]).lower().split()
        meteor = [ms.meteor_score(references=ref, hypothesis=hypo)]
        r = Rouge()
        rougel = [max([r.get_scores(output_text[0], c)[0]['rouge-l']['f'] for c in caption])]
    except ImportError:
        pass

    return [obj_iou, func_iou, rougel, meteor, pre, rec]


def prepare_coco_input(dataset_path):
    """Load COCO Caption minival dataset for evaluation.

    Expects:
        {dataset_path}/annotations/instances_minival2014.json
        {dataset_path}/annotations/captions_val2014.json
        {dataset_path}/image/{id}.jpg
        {dataset_path}/seg_label/{id}.png

    Returns:
        List of [image_path, prompt, captions, mask_path, category_dict].
    """
    seg_anno = json.load(open(os.path.join(dataset_path, 'annotations/instances_minival2014.json')))
    cap_anno = json.load(open(os.path.join(dataset_path, 'annotations/captions_val2014.json')))

    prompt = 'Write a one-sentence caption for this image:'

    cap_dic = {}
    for a in cap_anno['annotations']:
        cap_dic.setdefault(a['image_id'], []).append(a['caption'])

    input_data = []
    for img_info in seg_anno['images']:
        fn = str(img_info['id']).zfill(12)
        input_data.append([
            os.path.join(dataset_path, 'image', fn + '.jpg'),
            prompt,
            cap_dic.get(img_info['id'], ['']),
            os.path.join(dataset_path, 'seg_label', fn + '.png'),
            COCO_CATEGORIES,
        ])

    return input_data
