"""
전문상담교사 면담 축어록 텍스트마이닝 및 LDA 토픽모델링 분석
대구경북 지역 중등 전문상담교사 24명 대상 심리적 생존 경험 연구
"""

import os
import re
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ──────────────────────────────────────────────
def setup_korean_font():
    font_candidates = [
        'NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/nanum/NanumGothic.ttf',
    ]
    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams['font.family'] = 'NanumGothic'
            plt.rcParams['axes.unicode_minus'] = False
            return path
    # 폰트 없을 때 시스템 기본 한글 폰트 탐색
    for f in fm.findSystemFonts():
        if 'nanum' in f.lower() or 'malgun' in f.lower():
            fm.fontManager.addfont(f)
            plt.rcParams['font.family'] = fm.FontProperties(fname=f).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return f
    print("[경고] 한글 폰트를 찾지 못했습니다. 워드클라우드 폰트 경로를 직접 지정하세요.")
    return None

FONT_PATH = setup_korean_font()

# ── 불용어 사전 ─────────────────────────────────────────────────
STOPWORDS = set([
    '것', '수', '등', '연구', '논문', '하다', '있다', '이다', '되다',
    '하는', '하고', '에서', '에게', '으로', '에도', '에는', '에서는',
    '그리고', '하지만', '그래서', '그런데', '또한', '따라서',
    '때문', '경우', '통해', '위해', '대해', '관련', '통한',
    '이런', '이러한', '그런', '그러한', '어떤', '이', '그', '저',
    '선생님', '교사', '학교', '상담', '학생',  # 너무 일반적인 단어 제거
    '생각', '말씀', '이야기', '부분', '내용', '방법', '문제',
])

# ── 토픽별 레이블 및 성격 ────────────────────────────────────────
TOPIC_LABELS = {
    0: ('직무 스트레스와 소진', '위협 요인'),
    1: ('법적·제도적 취약성', '위협 요인'),
    2: ('수퍼비전과 사례 자문', '보호 요인(전문적)'),
    3: ('동료 지지 네트워크', '보호 요인(관계적)'),
    4: ('개인적 자기 돌봄', '보호 요인(개인적)'),
    5: ('지역적 특수성과 인프라 한계', '맥락적 조건'),
    6: ('의미 재구성과 성장', '성장 지향성'),
}


# ════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════

def load_texts(path: str) -> list[str]:
    """
    텍스트 파일을 로드한다.
    - 파일이 [참여자1], [참여자2] … 구분자를 포함하면 참여자별로 분리
    - 구분자가 없으면 단락(빈 줄 2개) 기준으로 분리
    - 파일이 없으면 샘플 데이터로 대체
    """
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # 참여자 구분자 패턴
        chunks = re.split(r'\[참여자\s*\d+\]', raw)
        chunks = [c.strip() for c in chunks if len(c.strip()) > 50]
        if len(chunks) >= 5:
            return chunks
        # 빈 줄 2개 이상 기준 단락 분리
        chunks = re.split(r'\n{2,}', raw)
        return [c.strip() for c in chunks if len(c.strip()) > 50]
    else:
        print(f"[안내] '{path}' 파일이 없습니다. 샘플 데이터로 실행합니다.")
        return _sample_texts()


def _sample_texts() -> list[str]:
    """연구 맥락에 맞는 샘플 면담 텍스트 (24명 시뮬레이션)"""
    samples = [
        "소진이 너무 심해서 더 이상 역할갈등을 버티기가 힘들었어요. 행정업무가 늘어나면서 감정노동도 극심해졌고 번아웃 직전이었습니다.",
        "자살위기 학생을 혼자 위기개입 해야 할 때 두려움이 엄청났어요. 업무과다에 스트레스가 쌓이다 보니 불면증과 두통까지 생겼습니다.",
        "비밀보장 원칙과 의무신고 사이에서 법적보호가 전혀 없다는 걸 깨달았어요. 악성민원에 교권침해까지 당하니 법률자문이 절실했습니다.",
        "상담기록이 법적분쟁에 증거로 쓰일까봐 개인정보 유출이 두려웠어요. 학부모항의와 소송위험이 현실이 되는 상황에서 보호법이 필요합니다.",
        "슈퍼비전을 받고 싶어도 비용부담이 너무 커요. 사례개념화나 역전이 관리를 제대로 배울 기회가 없으니 전문성이 정체되는 느낌입니다.",
        "집단수퍼비전이라도 받으려고 학회 사례발표에 참여했어요. 임상수퍼비전 받을 여건이 되면 피드백을 통해 훨씬 성장할 수 있을 텐데요.",
        "동료들끼리 카카오톡 단체방을 만들어서 정보공유를 해요. 소모임에서 경험공유를 하면 공감과 위로가 되어 스트레스가 줄어들더라고요.",
        "정기모임을 통한 학습공동체가 정말 중요한 지지원이에요. 전화상담으로라도 동료들과 소통하면서 힐링이 되고 견뎌낼 수 있었습니다.",
        "자기돌봄으로 명상과 운동을 꾸준히 해요. 경계설정과 마음챙김 덕분에 상담외시간차단이 가능해지고 워라밸을 유지할 수 있었습니다.",
        "요가와 호흡법, 저널링이 정서 조절에 도움이 돼요. 건강관리와 수면 관리를 철저히 하면서 자기이해도 깊어졌습니다.",
        "대구경북 지역은 순회상담을 해야 해서 이동피로가 심해요. 농어촌 지역 소규모학교는 인프라부족이 더 심각하고 전문가부족도 문제입니다.",
        "경북교육청의 지원이 아직 미흡하고 도시격차도 느껴져요. Wee클래스 시설부족과 예산부족으로 교육소외가 발생하고 소속감부재도 큽니다.",
        "소명의식을 가지고 일하면 어려운 상황에서도 버틸 수 있어요. 학생변화를 보면서 보람과 의미발견을 하고 외상후성장을 경험했습니다.",
        "직업만족과 자부심이 있어야 소진을 극복할 수 있어요. 감사와 감동, 성취감이 쌓이면서 가치관변화와 성숙을 경험하게 됩니다.",
        "역할갈등이 소진의 가장 큰 원인이에요. 교사이면서 상담자여야 하는 이중 역할에서 오는 업무과다와 감정노동이 너무 힘들었습니다.",
        "자살위기 개입 후에 심리적 충격이 컸어요. 대리외상과 공감피로가 쌓이면서 정서적고갈이 심해지고 무기력감이 찾아왔습니다.",
        "비밀보장 원칙을 지키다가 학교폭력 의무신고 압박을 받았어요. 상담일지가 증거로 요구되는 상황에서 면책특권의 필요성을 절감했습니다.",
        "악성민원으로 인한 교권침해가 정서적으로 너무 소모적이에요. 법적분쟁 위험성이 높아지면서 상담 기록 관리에 더욱 신경 쓰게 됩니다.",
        "수퍼비전 비용이 개인 부담이라 자주 받기 어려워요. 집단수퍼비전이라도 정기적으로 받을 수 있으면 사례개념화 능력이 향상될 것 같아요.",
        "역전이 감정을 혼자 처리하는 게 너무 힘들어요. 체계적인 임상수퍼비전이 있으면 전문적 성장과 소진 예방이 동시에 가능할 것입니다.",
        "카카오톡 단체방이 비공식 수퍼비전 역할을 해요. 동료들과의 경험공유와 정보공유가 공식 지원보다 더 실질적인 도움이 됩니다.",
        "소모임 학습공동체에서 공감과 위로를 받으면서 힘을 얻어요. 동료지지 네트워크가 소진 회복의 가장 중요한 자원이 된다고 생각합니다.",
        "명상과 마음챙김으로 자기돌봄을 실천해요. 경계설정을 통해 상담 시간 외에는 업무에서 분리되는 연습을 꾸준히 하고 있습니다.",
        "순회상담으로 여러 학교를 돌아다니다 보면 이동피로가 심해요. 대구경북 지역 특성상 지원 인프라가 부족해서 더욱 어려운 상황입니다.",
    ]
    return samples


# ════════════════════════════════════════════════════════════════
# 2. 형태소 분석 및 전처리
# ════════════════════════════════════════════════════════════════

def tokenize(texts: list[str], stopwords: set = STOPWORDS) -> tuple[list[list[str]], list[str]]:
    """
    각 문서에서 명사를 추출하여 토큰 리스트를 반환한다.
    Returns:
        tokenized_docs : 문서별 토큰 리스트
        all_tokens     : 전체 토큰 단순 병합 (빈도 분석용)
    """
    try:
        from konlpy.tag import Okt
        okt = Okt()
        tokenize_fn = lambda text: okt.nouns(text)
        print("[형태소 분석] KoNLPy Okt 사용")
    except ImportError:
        print("[형태소 분석] KoNLPy 미설치 → 공백/구두점 기반 간이 분리 사용")
        tokenize_fn = lambda text: re.findall(r'[가-힣]{2,}', text)

    tokenized_docs, all_tokens = [], []
    for doc in texts:
        tokens = [
            w for w in tokenize_fn(doc)
            if w not in stopwords and len(w) > 1
        ]
        tokenized_docs.append(tokens)
        all_tokens.extend(tokens)

    return tokenized_docs, all_tokens


# ════════════════════════════════════════════════════════════════
# 3. 빈도 분석 (표 5)
# ════════════════════════════════════════════════════════════════

# 대응 범주 사전
CATEGORY_MAP = {
    '소진': '위협 인식', '역할갈등': '위협 인식', '비밀보장': '위협 인식',
    '자살위기': '위협 인식', '행정업무': '위협 인식', '법적보호': '위협 인식',
    '교권침해': '위협 인식', '악성민원': '위협 인식', '업무과다': '위협 인식',
    '감정노동': '위협 인식', '대리외상': '위협 인식', '공감피로': '위협 인식',
    '역할모호성': '위협 인식', '학생자해': '위협 인식', '학교폭력': '위협 인식',
    '상담기록': '위협 인식', '연수부실': '위협 인식',
    '슈퍼비전': '관계적 지지 활용', '수퍼비전': '관계적 지지 활용',
    '동료지지': '관계적 지지 활용', '전문가자문': '관계적 지지 활용',
    '집단수퍼비전': '관계적 지지 활용', '사례개념화': '관계적 지지 활용',
    '동료상담': '관계적 지지 활용', '관리자교육': '관계적 지지 활용',
    '심리지원': '관계적 지지 활용',
    '자기돌봄': '자기보전 시도', '명상': '자기보전 시도', '운동': '자기보전 시도',
    '경계설정': '자기보전 시도', '마음챙김': '자기보전 시도',
    '회복탄력성': '자기보전 시도', '역전이': '자기보전 시도',
    '자기효능감': '자기보전 시도', '소진예방': '자기보전 시도',
    '외상후성장': '의미 재구성', '소명의식': '의미 재구성',
    '의미발견': '의미 재구성', '보람': '의미 재구성', '성장': '의미 재구성',
    '전문직정체성': '전문 정체성 유지', '역량강화': '전문 정체성 유지',
    '정체성혼란': '전문 정체성 유지',
    '대구경북': '지역적 특수성', '순회상담': '지역적 특수성',
    '농어촌': '지역적 특수성', '대구교육청': '지역적 특수성',
    '경북교육청': '지역적 특수성', '순회지원': '지역적 특수성',
}


def frequency_analysis(all_tokens: list[str], top_n: int = 50) -> pd.DataFrame:
    counter = Counter(all_tokens)
    top_words = counter.most_common(top_n)
    rows = []
    for rank, (word, freq) in enumerate(top_words, 1):
        rows.append({
            '순위': rank,
            '키워드': word,
            '빈도(회)': freq,
            '대응 범주': CATEGORY_MAP.get(word, '-'),
        })
    return pd.DataFrame(rows)


def save_frequency_table(df: pd.DataFrame, path: str = 'results/표5_상위키워드50.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"[저장] 빈도표 → {path}")
    print(df.to_string(index=False))


# ════════════════════════════════════════════════════════════════
# 4. 워드클라우드
# ════════════════════════════════════════════════════════════════

def generate_wordcloud(all_tokens: list[str], font_path: str = None,
                       save_path: str = 'results/wordcloud.png'):
    from wordcloud import WordCloud

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    freq = dict(Counter(all_tokens))

    wc_kwargs = dict(
        width=1200, height=800,
        background_color='white',
        max_words=100,
        colormap='viridis',
        prefer_horizontal=0.7,
    )
    if font_path and os.path.exists(font_path):
        wc_kwargs['font_path'] = font_path

    wc = WordCloud(**wc_kwargs).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('전문상담교사 면담 축어록 주요 키워드', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] 워드클라우드 → {save_path}")


# ════════════════════════════════════════════════════════════════
# 5. LDA 모델링
# ════════════════════════════════════════════════════════════════

def build_lda_model(tokenized_docs: list[list[str]],
                    num_topics: int = 7,
                    passes: int = 20,
                    iterations: int = 400,
                    random_state: int = 42):
    from gensim import corpora, models

    # 사전 및 코퍼스 생성
    dictionary = corpora.Dictionary(tokenized_docs)
    # 너무 희귀하거나 너무 흔한 단어 제거
    dictionary.filter_extremes(no_below=2, no_above=0.85)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        random_state=random_state,
        alpha='auto',
        eta='auto',
        per_word_topics=True,
    )
    return lda, corpus, dictionary


def find_optimal_topics(tokenized_docs: list[list[str]],
                        topic_range: range = range(4, 11),
                        save_path: str = 'results/토픽수_최적화.png') -> int:
    """Coherence score(c_v)로 최적 토픽 수 탐색"""
    from gensim import corpora, models
    from gensim.models.coherencemodel import CoherenceModel

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.85)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    coherence_scores = []
    for k in topic_range:
        lda = models.LdaModel(
            corpus=corpus, id2word=dictionary,
            num_topics=k, passes=15, random_state=42,
        )
        cm = CoherenceModel(model=lda, texts=tokenized_docs,
                            dictionary=dictionary, coherence='c_v')
        coherence_scores.append(cm.get_coherence())
        print(f"  K={k:2d}  Coherence={coherence_scores[-1]:.4f}")

    best_k = list(topic_range)[coherence_scores.index(max(coherence_scores))]

    # 시각화
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(topic_range), coherence_scores, 'o-', color='steelblue', lw=2)
    ax.axvline(best_k, color='tomato', linestyle='--', label=f'최적 K={best_k}')
    ax.set_xlabel('토픽 수 (K)', fontsize=12)
    ax.set_ylabel('Coherence Score (c_v)', fontsize=12)
    ax.set_title('LDA 최적 토픽 수 탐색', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] 토픽 수 최적화 그래프 → {save_path}  (최적 K={best_k})")
    return best_k


# ════════════════════════════════════════════════════════════════
# 6. 결과 시각화
# ════════════════════════════════════════════════════════════════

def plot_topic_keywords(lda, num_topics: int = 7, top_n: int = 20,
                        save_path: str = 'results/토픽별_키워드.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cols = 2
    rows = (num_topics + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 5))
    axes = axes.flatten()

    for t in range(num_topics):
        words_weights = lda.show_topic(t, topn=top_n)
        words = [w for w, _ in words_weights]
        weights = [s for _, s in words_weights]
        label, nature = TOPIC_LABELS.get(t, (f'Topic {t+1}', ''))
        color = '#d9534f' if '위협' in nature else '#5cb85c' if '보호' in nature \
            else '#f0ad4e' if '맥락' in nature else '#5bc0de'

        ax = axes[t]
        bars = ax.barh(range(len(words)), weights[::-1], color=color, alpha=0.8)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words[::-1], fontsize=9)
        ax.set_title(f'Topic {t+1}: {label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('β 가중치', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    # 빈 subplot 제거
    for i in range(num_topics, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('LDA 토픽별 상위 키워드 및 가중치', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] 토픽별 키워드 그래프 → {save_path}")


def plot_document_topic_heatmap(lda, corpus, num_topics: int = 7,
                                 save_path: str = 'results/문서_토픽_히트맵.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    doc_topic_matrix = np.zeros((len(corpus), num_topics))
    for i, bow in enumerate(corpus):
        for t, prob in lda.get_document_topics(bow, minimum_probability=0):
            doc_topic_matrix[i, t] = prob

    fig, ax = plt.subplots(figsize=(12, max(6, len(corpus) * 0.4)))
    im = ax.imshow(doc_topic_matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label='토픽 확률')
    ax.set_xticks(range(num_topics))
    ax.set_xticklabels(
        [f'T{t+1}\n{TOPIC_LABELS.get(t, (f"Topic{t+1}",""))[0][:8]}' for t in range(num_topics)],
        fontsize=8,
    )
    ax.set_yticks(range(len(corpus)))
    ax.set_yticklabels([f'참여자 {i+1:02d}' for i in range(len(corpus))], fontsize=8)
    ax.set_title('참여자별 토픽 분포 히트맵', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] 문서-토픽 히트맵 → {save_path}")


def plot_topic_distribution(lda, corpus, num_topics: int = 7,
                             save_path: str = 'results/토픽_분포_비율.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    topic_totals = np.zeros(num_topics)
    for bow in corpus:
        for t, prob in lda.get_document_topics(bow, minimum_probability=0):
            topic_totals[t] += prob

    labels = [f'T{t+1}: {TOPIC_LABELS.get(t, (f"Topic{t+1}",""))[0]}' for t in range(num_topics)]
    colors = ['#d9534f', '#e67e22', '#3498db', '#2ecc71', '#27ae60', '#f39c12', '#9b59b6']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 파이 차트
    wedges, texts, autotexts = ax1.pie(
        topic_totals, labels=None, autopct='%1.1f%%',
        colors=colors, startangle=140, pctdistance=0.8,
    )
    ax1.legend(wedges, labels, loc='lower left', fontsize=8, bbox_to_anchor=(-0.2, -0.15))
    ax1.set_title('토픽별 전체 비중', fontsize=13, fontweight='bold')

    # 막대 차트
    bars = ax2.bar(range(num_topics), topic_totals, color=colors, alpha=0.85, edgecolor='white')
    ax2.set_xticks(range(num_topics))
    ax2.set_xticklabels([f'T{t+1}' for t in range(num_topics)], fontsize=10)
    ax2.set_ylabel('누적 토픽 확률 합계', fontsize=11)
    ax2.set_title('토픽별 누적 확률 합계', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, topic_totals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('LDA 토픽 분포 분석', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] 토픽 분포 그래프 → {save_path}")


def save_topic_keyword_table(lda, num_topics: int = 7, top_n: int = 20,
                              save_path: str = 'results/표A1_토픽별_키워드.csv'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rows = []
    for t in range(num_topics):
        label, nature = TOPIC_LABELS.get(t, (f'Topic {t+1}', ''))
        for rank, (word, weight) in enumerate(lda.show_topic(t, topn=top_n), 1):
            rows.append({
                '토픽번호': t + 1,
                '토픽명': label,
                '성격': nature,
                '순위': rank,
                '키워드': word,
                'β 가중치': round(weight, 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[저장] 토픽별 키워드표 → {save_path}")
    return df


# ════════════════════════════════════════════════════════════════
# 7. pyLDAvis 인터랙티브 시각화 (선택)
# ════════════════════════════════════════════════════════════════

def save_pyldavis(lda, corpus, dictionary,
                  save_path: str = 'results/lda_visualization.html'):
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pyLDAvis.enable_notebook()
        vis = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(vis, save_path)
        print(f"[저장] pyLDAvis 인터랙티브 시각화 → {save_path}")
    except ImportError:
        print("[건너뜀] pyLDAvis 미설치 → pip install pyldavis")
    except Exception as e:
        print(f"[건너뜀] pyLDAvis 오류: {e}")


# ════════════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════════════

def main(
    data_path: str = 'interview_all.txt',
    num_topics: int = 7,
    optimize_k: bool = False,
    top_keywords: int = 50,
):
    print("=" * 60)
    print("  전문상담교사 심리적 생존 경험 - LDA 토픽모델링 분석")
    print("=" * 60)

    # 1. 데이터 로드
    texts = load_texts(data_path)
    print(f"\n[데이터] 총 {len(texts)}개 문서 로드 완료")

    # 2. 형태소 분석
    print("\n[전처리] 형태소 분석 중...")
    tokenized_docs, all_tokens = tokenize(texts)
    print(f"  전체 토큰 수: {len(all_tokens):,}개 | 고유 토큰: {len(set(all_tokens)):,}개")

    # 3. 빈도 분석 (표 5)
    print(f"\n[빈도 분석] 상위 {top_keywords}개 키워드 추출 중...")
    freq_df = frequency_analysis(all_tokens, top_n=top_keywords)
    save_frequency_table(freq_df, 'results/표5_상위키워드50.csv')

    # 4. 워드클라우드
    print("\n[워드클라우드] 생성 중...")
    try:
        generate_wordcloud(all_tokens, font_path=FONT_PATH)
    except ImportError:
        print("[건너뜀] wordcloud 미설치 → pip install wordcloud")

    # 5. 최적 토픽 수 탐색 (선택적)
    if optimize_k:
        print("\n[최적화] 최적 토픽 수 탐색 중...")
        try:
            num_topics = find_optimal_topics(tokenized_docs)
        except Exception as e:
            print(f"  최적화 실패: {e} → 기본값 K={num_topics} 사용")

    # 6. LDA 모델 학습
    print(f"\n[LDA] K={num_topics} 토픽 모델 학습 중...")
    try:
        from gensim import corpora, models
        lda, corpus, dictionary = build_lda_model(tokenized_docs, num_topics=num_topics)

        print(f"\n  Perplexity: {lda.log_perplexity(corpus):.4f}")

        # 7. 결과 저장
        print("\n[시각화] 결과 생성 중...")
        plot_topic_keywords(lda, num_topics=num_topics)
        plot_document_topic_heatmap(lda, corpus, num_topics=num_topics)
        plot_topic_distribution(lda, corpus, num_topics=num_topics)
        save_topic_keyword_table(lda, num_topics=num_topics)
        save_pyldavis(lda, corpus, dictionary)

        # 모델 저장
        os.makedirs('results/model', exist_ok=True)
        lda.save('results/model/lda_model')
        dictionary.save('results/model/dictionary')
        print("[저장] LDA 모델 → results/model/")

    except ImportError:
        print("[건너뜀] gensim 미설치 → pip install gensim")

    print("\n" + "=" * 60)
    print("  분석 완료. 결과 파일: results/ 폴더 확인")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LDA 토픽모델링 분석')
    parser.add_argument('--data', default='interview_all.txt', help='면담 축어록 파일 경로')
    parser.add_argument('--topics', type=int, default=7, help='토픽 수 (기본: 7)')
    parser.add_argument('--optimize', action='store_true', help='최적 토픽 수 자동 탐색')
    parser.add_argument('--top_n', type=int, default=50, help='상위 키워드 수 (기본: 50)')
    args = parser.parse_args()

    main(
        data_path=args.data,
        num_topics=args.topics,
        optimize_k=args.optimize,
        top_keywords=args.top_n,
    )
