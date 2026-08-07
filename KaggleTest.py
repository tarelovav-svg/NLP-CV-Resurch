# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:34:25 2026

@author: Александр
"""

import re
import nltk
import pymorphy3
from navec import Navec
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


from functools import lru_cache
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df1 = pd.read_excel("test_dataset_test_correct.xlsx")

df1.loc[df1['group'] == 'Благоустройство и комфортная городская среда', 'label'] = 0
df1.loc[df1['group'] == 'Улично-дорожная сеть и транспорт', 'label'] = 1
df1.loc[df1['group'] == 'Коммунальные ресурсы', 'label'] = 2
df1.loc[df1['group'] == 'Многоквартирные дома и управляющие компании', 'label'] = 3
df1.loc[df1['group'] == 'Социальная сфера', 'label'] = 4

df = df1.drop(['group'], axis=1)

df['text'] = df.apply(lambda row: row['text1'], axis=1)
print(df.head())
print(df.info())
print(df.duplicated().sum())
print(df.isna().sum())

# Предварительная обработка основного текста: строчные буквы, удаление некириллических символов, разметка

df['text_clean'] = df['text'].str.lower()
df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(r'[^а-яё\s]', '', x))
df['tokens'] = df['text_clean'].apply(lambda x: x.split())
print(df['tokens'])

# Здесь мы удаляем стоп-слова, а также отфильтровываем 75 наиболее часто встречающихся слов. 
# Это полезно для предварительного анализа (облака слов, топ-маркеры и т.д.),
# но мы не удаляем их для модели, поскольку стоп-слова и часто встречающиеся слова 
# (например, "не", "ясно") может содержать важную информацию о настроениях.

nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))
df['tokens_nostop'] = df['tokens'].apply(lambda x: [word for word in x if word not in russian_stopwords])
all_tokens = [word for tokens in df['tokens_nostop'] for word in tokens]
freqs = Counter(all_tokens)
most_common_words = {w for w, _ in freqs.most_common(75)}
df['tokens_nostop'] = df['tokens_nostop'].apply(
    lambda tokens: [word for word in tokens if word not in most_common_words]
)

# Лемматизация с помощью pymorphy3

tqdm.pandas()   
morph = pymorphy3.MorphAnalyzer()

@lru_cache(maxsize=100000)
def lemmatize_word(word):
    return morph.parse(word)[0].normal_form

def lemmatize_tokens(tokens):
    return [lemmatize_word(word) for word in tokens]

df['tokens_lemma'] = df['tokens'].progress_apply(lemmatize_tokens)
print(df[['text', 'tokens', 'tokens_nostop', 'tokens_lemma']].head())

# разведочный анализ данных

label_names = {0: 'Благоустройство', 1: 'Дорожная сеть', 2: 'Коммунальные', 3: 'МКД', 4: 'Социальная сфера'}

df['label'].map(label_names).value_counts().plot(
    kind='bar'
)
plt.title("Распределение по классам")
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.xticks(rotation=0)
plt.show()

# Облака слов по классам заявок

for label in [0, 1, 2, 3, 4]:
    tokens_list = df[df['label'] == label]['tokens_nostop']
    all_text = ' '.join([' '.join(tokens) for tokens in tokens_list])
    
    wordcloud = WordCloud(
        width=800, height=400, background_color='white',
        collocations=False, 
    ).generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Облако слов — {label_names[label]}")
    plt.show()

# Лучшие биграммы по классу заявок

for label in [0, 1, 2, 3, 4]:
    tokens_list = df[df['label'] == label]['tokens_nostop']
    bigrams = []
    for tokens in tokens_list:
        bigrams.extend(ngrams(tokens, 2))
    bigram_counts = Counter(bigrams).most_common(15)
    labels = [' '.join(gram) for gram, _ in bigram_counts]
    counts = [count for _, count in bigram_counts]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)
    plt.xticks(rotation=45)
    plt.title(f'{label_names[label]} Bigrams (2-grams)')
    plt.show()    
    
# Загрузка предварительно подготовленных вставок слов (Navec, 300d)
# Эти вложения обучены на российских корпусах и обеспечивают плотный вектор 
# представления для слов. Мы будем использовать их в качестве инициализации для нашей модели TextCNN.

#!wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar -q

navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

print("Embedding dimension:", navec.pq.dim)
print("Example vector for 'кот':", navec['кот'][:10])

# Построение словаря и матрицы встраивания (векторы PAD/UNK + Navec)
PAD_IDX, UNK_IDX = 0, 1
words = list(navec.vocab.words)

word2idx = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
word2idx.update({w: i + 2 for i, w in enumerate(words)})

embedding_dim = navec.pq.dim
embedding_matrix = torch.zeros((len(word2idx), embedding_dim))

embedding_matrix[UNK_IDX] = torch.mean(
    torch.stack([torch.tensor(navec[w]) for w in words]), dim=0
)

for w, i in word2idx.items():
    if w not in ["<PAD>", "<UNK>"]:
        embedding_matrix[i] = torch.tensor(navec[w])
        
# Преобразовать токены в индексные последовательности фиксированной длины (дополнить/усечь до MAX_LEN)
MAX_LEN = 100

def tokens_to_seq(tokens, max_len=MAX_LEN):
    seq = [word2idx.get(tok, UNK_IDX) for tok in tokens]
    if len(seq) < max_len:
        seq += [PAD_IDX] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

X = [tokens_to_seq(toks) for toks in df["tokens_lemma"]]
y = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# Набор данных Torch и загрузчик данных (для группирования и перетасовки)
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=256, shuffle=True)
test_loader  = DataLoader(TextDataset(X_test, y_test), batch_size=256)

# Модель TextCNN для классификации предложений
# - Слой встраивания, инициализированный предварительно подготовленными векторами Navec (может быть заморожен / разморожена разморозка)
# - Несколько блоков Conv1D с разными размерами ядра (2,3,4,5) для захвата n-граммовых объектов
# - Повторная активация + локальное отключение после каждого обновления для уменьшения переобучения
# - Глобальное максимальное объединение с течением времени для получения функций фиксированного размера
# - Окончательное отключение + полностью подключенный уровень → логиты классов

class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, kernel_sizes=[2,3,4,5], num_filters=128, dropout=0.1):
        super(TextCNN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True, padding_idx=0
        )
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_filters, (k, embedding_dim)),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)          # (batch, seq_len, emb_dim)
        x = x.unsqueeze(1)             # (batch, 1, seq_len, emb_dim)

        pooled_outs = []
        for conv in self.convs:
            c = conv(x).squeeze(3)                     # (batch, num_filters, seq_len-k+1)
            mp = F.max_pool1d(c, c.size(2)).squeeze(2) # (batch, num_filters)
            pooled_outs.append(mp)

        cat = torch.cat(pooled_outs, dim=1)            # (batch, num_filters*len(kernels))
        cat = self.dropout(cat)
        
        return self.fc(cat)


num_classes = len(set(y))

#num_classes = 4
print(num_classes)
model = TextCNN(embedding_matrix, num_classes)

# Устройство настройки и оптимизатор
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)
optimizer = optim.AdamW(model.parameters(),lr=2e-3, weight_decay=1e-5)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Цикл обучения + валидации
EPOCHS = 3
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    
    for X_batch, y_batch in train_bar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
        train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_val_loss = val_loss / len(test_loader)
    val_acc = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f} | "
          f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

# Постройте кривые потерь и точности
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Потери на тренировочных данных")
plt.plot(val_losses, label="Потери на валидационных данных")
plt.xlabel("Эпохи")
plt.ylabel("Потери")
plt.title("Кривые потерь")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Точность на тренировочных данных")
plt.plot(val_accs, label="точность на валидационных данных")
plt.xlabel("Эпохи")
plt.ylabel("Точность")
plt.title("Кривые точности")
plt.legend()

plt.show()

# Окончательные показатели по набору для проверки
print("\nClassification Report (Validation):")
print(classification_report(all_labels, all_preds, digits=3))

# Предварительная обработка для вывода (очистка, токенизация, лемматизация, дополнение/усечение)
def preprocess_text(text, max_len=MAX_LEN):
    text = text.lower()                                # lowercase
    text = re.sub(r'[^а-яё\s]', '', text)              # remove non-Cyrillic chars
    tokens = text.split()                              # tokenize
    tokens = [lemmatize_word(w) for w in tokens]       # lemmatize
    seq = tokens_to_seq(tokens, max_len)               # to indices with pad/truncate
    return torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)

# Функция прогнозирования (получение идентификатора класса из модели)
def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        seq = preprocess_text(text)
        output = model(seq)
        pred = output.argmax(dim=1).item()
    return pred

# Отзывы для тестирования
reviews = [
"Много мусора на тротуаре около автобусной остановки. ",
"Доброго времени суток! Прошу обратить внимание на повторное образование глубоких ям на проезжей части дороги по улице Урожайной , в районе домов с 83 по 109. Просим принять меры по ремонту дороги , как вы ранее и обещали. ", 
"Очень маленькая автостоянка для автомобилей пациентов медицинского учреждения - поликлиники. Машину приходится парковать вне стоянки под угрозой её эвакуации на штрафстоянку.Достаточно всего лишь передвинуть вазоны с растениями и автостоянка увеличится в 4 раза.Рассмотрите пожалуйста возможность увеличения парковки для пациентов поликлиники по адресу: Олимпийский проспект, строение 38. ",
"прошу поправить дерево на набережной Мзымта", 
"у взрослой поликлиники 8 парковочных мест! в поликлиники более 8 тысяч человек! как оставлять машину и где? в поликлинику приезжают инвалиды и за неотложной помощью! почему задача сотрудников гбдд забирать машину у поликлиники? как туда ходить!! почему гбдд охотиться за машинами у поликлиник?!сделайте парковку для людей ! а не гуляющих по морю!!! ",
"Прошу дать разъяснения возможно ли установление световой вывески заведения общепита НА Английском Языке на территории ФТ Сириус и порядок согласования установления вывесок заведений общепита .И если существует запрет, то на основании какого нормативного акта и какая ответственность придусмотрена. ",
"Ночью с 13 на 14 февраля, во время сильного ветра с крыши дома сорвало металическую обшивку с трубы. Труба кирпичная, она разваливается. Дом находиться на проезжей и пешеходный части, один из главных проходов к набережной. В любое время может с крыши упасть обшивка и сама труба. Просим, принять меры для устранения аварийной ситуации. ", 
"Повреждён тротуар. во время дождя образуется огромная лужа и приходится к остановке общественного транспорта обходить по грязи. ",
"Здравствуйте! Просьба проверить законность установки забора и здания, которые перегораживают тротуар. размер тротуара явно заужен и не соответствует нормам безопасности и доступности среды. ", 
"Здравствуйте! просьба заделать яму в асфальте у ж/д переезда на ул. Веселой. ", 
"По ул .Луначарского  дом 7 после ремонтных работ,  до сих пор не вычищена сточная канава.   Которая засыпана булыжниками , мусором, корягами деревьев. При дожде!!! затапливает дворовую территорию. так как воде из за мусора некуда деваться. Прошу принять меры. И привести в надлежащий вид. "
"Прошу заменить песок на детской площадке, песка практически не осталось, а также в нем огромное количество технический вкраплений (синтетическая ткань) ",
"Нарушение пдд , стоянка автомобилей в зоне действия знака 3.27 ( остановка запрещена) ",
"Добрый день, после проведения ремонтных работ по ул. Луносармкого 7,  строителная бригада оставила после себя глыбы камней, мусор,  невычещенная сливная канава, пни . На данный момент весь двор стоит в воде, так как срочная канава закидали камнями и илом. Невозможно припарковать личный автомобиль около дома. Прошу как можно скорее устранить дануу проблему. До начала проведения ремонтных работ, данная территория была благоустроена своими силами. ", 
"Прошу обратить внимание на необходимость уборки территории выхода с Бульвара надежд. камни, сломанные ограждения, мусор и сухая трава. ", 
"необходимо поправить дорожный знак на улице Бульвар Надежд",
"Прошу обратить внимание на эти участки тротуаров вдоль улицы Чемпионов. В мой сириус нет информации о проводимых работах, отсутствует информация о ремонте и на самих участках. это разрытие с 2025 года. пора закончить работы. ", 
"прошу обратить внимание, что газон перепахан автомобилями, мотоциклами, используется для проезда на парковку. необходимо принятие мер по предотвращению использования данного участка для проезда. ", 
"Прошу промыть контейнеры и площадку под контейнерами, а также оборудовать контейнерный бункер согласно стандарту всех контейнерных площадок в Сириусе. ",
"В доме 31/3 по улице Урожайной не слышны сирены при угрозах БПЛА и непогоды. Из-за того, что интернет отключают даже смс не всегда приходит. Поэтому если что-то случается серьезное (работа ПВО, например), узнаём уже по взрывам. Помогите с проблемой установки громкоговорителей в районе села Весёлого. ", 
"От ветра наклонился кипарис и может повалиться в любой момент. ", 
"Добрый день, мной были направлены два обращения №742 и 1001 по вопросу кронированию платанов на улице Веселой . Работы так и не были произведены. Просьба привести деревья в порядок. Последний ваш ответ копирую ниже:Добрый день! Благодарим за бдительность и сообщаем, что при подготовке к кронированию указанных деревьев выяснились дополнительные обстоятельства, в результате чего работы были приостановлены. В данный момент кронирование ветвей указанных платанов не представляется возможным в связи с пролеганием воздушной линии электропередач. По информации балансодержателя АО Россети Кубань, демонтаж крепления ЛЭП к деревьям запланирован в период до 31.10.2025 года. После проведения данных работ кронирование будет выполнено.",
"Зачем понятно, но для кого делаются спортивные площадки на бульваре, если на территории Бархатные сезоны( с лева) и территории Бридж( с права) уже есть масса детских, спортивных, развлекательных площадок, а желых домов в радиусе 2 км нет.?! Вернее есть два двух этажных(одноподъездных) дома и 10 частных домов, но у них уже есть, пусть и  Советская, но детско-спортивная площадка, у моря! ",
"Прошу принять меры реагирования по отлову безнадзорной стаи крупных собак на ул. Акаций. Район 1КПП ЖК Фрукты, также могут обитать в районе мусорки у частных домов (выезд с ул. Акаций на трассу). ,В радиусе 1000 метров нет современных детских, спортивных площадок! Когда планируется решение данного вопроса? Спасибо.",
"обратите внимание на пальму возле отеля Мантера резорт. необходимо или удалить или принять меры по обработке", 
"Здравствуйте! Прошу оказать содействие в обеспечении безопасности территории между Сигмой Б и Сигмой А, по улице Рекордов, от стаи бездомных собак. Кидаются на детей, велосипеды. Спасибо большое! ",
"На опорах освещения в олимпийском парке висят таблички о мерах безопасности со времен ковида. Сейчас они не актуальны и находятся в печальном состоянии. ",
"УК не отчитывается перед собственниками по своим услугам и тратам, собрание не проводит."
]

for i, review in enumerate(reviews, 1):
    pred = predict_sentiment(review)
    print(f"{i}. {label_names[pred]} → {review[:120]}...")