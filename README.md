## Project Overview

Video game digital distribution platform seperti Steam memiliki katalog yang sangat luas sehingga membuat pengguna sering kesulitan menemukan game yang sesuai dengan minat dan kebiasaan bermain mereka.

Dengan lebih dari 41 juta ulasan dan interaksi pengguna yang terekam secara eksplisit (termasuk apakah pengguna merekomendasikan suatu game, lama bermain, dan tanggal ulasan), analisis data ini sangat berpotensi membantu membangun sistem rekomendasi yang dapat mempersonalisasi tawaran game kepada setiap pengguna.

Sistem rekomendasi tidak hanya meningkatkan kepuasan pengguna dengan menampilkan game yang relevan, tetapi juga berdampak langsung pada peningkatan retensi dan pengeluaran pengguna di platform. Oleh karena itu, dalam proyek ini dikembangkan sistem rekomendasi berbasis deep learning yang memanfaatkan interaksi antara pengguna dan game serta fitur-fitur permainan untuk memberikan Top-N rekomendasi bagi setiap pengguna platform Steam.


## Business Understanding

### Problem Statements
- Bagaimana merancang sistem rekomendasi yang mampu memprediksi game baru yang akan disukai pengguna berdasarkan data riwayat interaksi pengguna dan metadata game?
- Bagaimana mengevaluasi bahwa sistem rekomendasi benar-benar merekomendasikan konten yang sesuai dengan preferensi dan pengalaman dari pengguna?

### Goals
- Mengembangkan model deep learning (Neural Collaborative Filtering) yang merepresentasikan preferensi pengguna dan fitur game dalam satu ruang vektor.
- Mengevaluasi performa sistem rekomendasi menggunakan metrik klasifikasi (Akurasi, Presisi, Recall, F1) terhadap variabel `is_recommended`.


## Data Understanding

Dataset yang digunakan dalam proyek ini berisi informasi mengenai interaksi pengguna dengan game di platform distribusi digital Steam serta metadata dari masing-masing game tersebut. Data ini sangat relevan untuk membangun sistem rekomendasi karena mencerminkan preferensi eksplisit pengguna terhadap berbagai judul game dan menyediakan deskripsi mengenai konten game. Data ini tersedia di platform Kaggle Dataset dengan nama [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam).

Dataset terdiri dari dua file utama yaitu:
1. `games.csv` memuat metadata game. File ini berisi informasi mendetail tentang game-game yang tersedia di Steam dengan total 50.872 baris dan 13 kolom, mencakup informasi sebagai berikut:

    | Kolom            | Tipe Data | Deskripsi                                                       |
    | ---------------- | --------- | --------------------------------------------------------------- |
    | `app_id`         | int64     | ID unik untuk setiap game                                       |
    | `title`          | object    | Judul atau nama game                                            |
    | `date_release`   | object    | Tanggal rilis game                                              |
    | `win`            | bool      | Apakah game tersedia untuk platform Windows (True/False)        |
    | `mac`            | bool      | Apakah game tersedia untuk platform Mac (True/False)            |
    | `linux`          | bool      | Apakah game tersedia untuk platform Linux (True/False)          |
    | `rating`         | object    | Rating ulasan pengguna (seperti 'Very Positive', 'Mixed', dll.) |
    | `positive_ratio` | int64     | Persentase ulasan positif pengguna terhadap total ulasan        |
    | `user_reviews`   | int64     | Total jumlah ulasan dari pengguna                               |
    | `price_final`    | float64   | Harga akhir game setelah diskon (dalam satuan USD)              |
    | `price_original` | float64   | Harga asli sebelum diskon                                       |
    | `discount`       | float64   | Besarnya diskon dalam bentuk desimal
    | `steam_deck`     | bool      | Apakah game kompatibel dengan Steam Deck (True/False)           |

2. `recommendations.csv` mencatat interaksi eksplisit antara pengguna dan game dalam bentuk ulasan atau review. Total terdapat lebih dari 41 juta baris dan memiliki 8 kolom, yaitu sebagai berikut:

    | Kolom            | Tipe Data | Deskripsi                                                           |
    | ---------------- | --------- | ------------------------------------------------------------------- |
    | `app_id`         | int64     | ID game yang diulas                                                 |
    | `helpful`        | int64     | Jumlah pengguna lain yang menganggap ulasan ini membantu            |
    | `funny`          | int64     | Jumlah pengguna lain yang menandai ulasan ini lucu                  |
    | `date`           | object    | Tanggal ulasan ditulis                                              |
    | `is_recommended` | bool      | Apakah pengguna merekomendasikan game (True/False)                  |
    | `hours`          | float64   | Total jam bermain yang dilaporkan pengguna pada saat menulis ulasan |
    | `user_id`        | int64     | ID pengguna yang menulis ulasan                                     |
    | `review_id`      | int64     | ID unik untuk setiap ulasan (review)                                |


## Data Preparation
Proses persiapan data dilakukan secara berurutan sebagai berikut:

### Inisialisasi dan Label Encoding
Untuk memudahkan pemrosesan maka dilakukan encoding pada beberapa kolom menggunakan LabelEncoder. Tahapan ini bertujuan untuk mengubah ID pengguna, ID game, dan rating dari format asli menjadi indeks integer bertipe `int64` agar dapat digunakan pada layer embedding di model deep learning.
```
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder()
item_enc = LabelEncoder()
rating_enc = LabelEncoder()

user_enc.fit(recommendations['user_id'].unique())
item_enc.fit(games['app_id'].unique())
rating_enc.fit(games['rating'].unique())
```

### Sampling dan Penggabungan Data
Karena ukuran data yang besar (lebih dari 41 juta baris), maka diambil 500.000 sampel secara acak agar proses pelatihan lebih efisien. Data interaksi kemudian digabung dengan metadata game menggunakan kolom `app_id`.
```
import pandas as pd

df_sampled = recommendations.sample(n=500_000, random_state=42).reset_index(drop=True)
df_merged = pd.merge(df_sampled, games, on='app_id', how='left')
```

### Penanganan Nilai Kosong
Pada bagian ini penting untuk memastikan bahwa tidak ada nilai kosong yang masuk ke dalam model karena bisa menyebabkan error atau ketidakkonsistenan saat pelatihan model. Nilai kosong pada fitur boolean diisi dengan `False`, pada kategori diisi dengan label "`Unknown`", dan pada numerik diisi dengan rata-rata global dari dataset game.
```
for col in ['win', 'mac', 'linux', 'steam_deck']:
    df_merged[col].fillna(False, inplace=True)

df_merged['rating'].fillna('Unknown', inplace=True)

for col in ['positive_ratio', 'user_reviews', 'price_final']:
    df_merged[col].fillna(games[col].mean(), inplace=True)
    
df_merged.dropna(subset=['app_id', 'user_id', 'is_recommended'], inplace=True)
```

### Encoding dan Scaling
Encoding diperlukan untuk mengubah variabel kategorikal menjadi angka. Lalu scaling atau standarisasi dilakukan pada fitur numerik agar model lebih cepat konvergen dan tidak berat sebelah ketika dalam proses pelatihan.
```
from sklearn.preprocessing import StandardScaler

# Encoding
df_merged['user_encoded'] = user_enc.transform(df_merged['user_id'])
df_merged['item_encoded'] = item_enc.transform(df_merged['app_id'])
df_merged['rating_encoded'] = rating_enc.transform(df_merged['rating'])

# Scaling
scaler = StandardScaler()
scaler.fit(games[['positive_ratio', 'user_reviews', 'price_final']])
df_merged[['positive_ratio', 'user_reviews', 'price_final']] = scaler.transform(
    df_merged[['positive_ratio', 'user_reviews', 'price_final']])
```
Setelah melakukan imputasi, encoding, dan scaling, kemudian menyiapkan delapan fitur konten game yang akan digabungkan dengan embedding pengguna dan game di model. Ketiga fitur numerik seperti `positive_ratio`, `user_reviews`, dan `price_final` telah distandarisasi menggunakan StandardScaler. Kolom `rating` diubah menjadi `rating_encoded` (LabelEncoder). Empat kolom boolean yaitu `win`, `mac`, `linux`, dan `steam_deck` digunakan langsung sebagai indikator kompatibilitas platform. Semua delapan nilai ini kemudian dibangun menjadi satu tensor `all_item_features_tensor` (ukuran num_items × 8) untuk input ke jalur MLP pada model.

### Pembangunan Tensor Fitur Game
Model rekomendasi akan menggabungkan fitur konten game dalam proses prediksi. Oleh karena itu, akan dibuat tensor yang menyimpan fitur tiap game. Fitur ini disusun dalam urutan yaitu numerik → rating → platform.
```
import torch

processed_games = games.copy()
for col in ['win', 'mac', 'linux', 'steam_deck']:
    processed_games[col].fillna(False, inplace=True)
processed_games['rating'].fillna('Unknown', inplace=True)
for col in ['positive_ratio', 'user_reviews', 'price_final']:
    processed_games[col].fillna(processed_games[col].mean(), inplace=True)
processed_games[['positive_ratio', 'user_reviews', 'price_final']] = scaler.transform(
    processed_games[['positive_ratio', 'user_reviews', 'price_final']])
processed_games['rating_encoded'] = rating_enc.transform(processed_games['rating'])

num_features = 3 + 1 + 4  # 3 numerik + rating + 4 platform
n_items_overall = len(item_enc.classes_)
all_item_features_tensor = torch.zeros(n_items_overall, num_features, dtype=torch.float32)

for _, row in processed_games.iterrows():
    item_id = item_enc.transform([row['app_id']])[0]
    features = row[['positive_ratio', 'user_reviews', 'price_final', 'rating_encoded', 'win', 'mac', 'linux', 'steam_deck']].astype(np.float32).values
    all_item_features_tensor[item_id] = torch.tensor(features)
```

### Membuat Dataset PyTorch
Untuk memudahkan pelatihan model dengan PyTorch, maka akan dibuat kelas dataset khusus. Dataset ini akan memberikan batch berupa user ID, item ID, fitur konten game, dan label (`is_recommended`).
```
from torch.utils.data import Dataset

class SteamDatasetWithFeatures(Dataset):
    def __init__(self, df_data, all_item_features_tensor):
        self.users = torch.tensor(df_data['user_encoded'].values, dtype=torch.long)
        self.items = torch.tensor(df_data['item_encoded'].values, dtype=torch.long)
        self.labels = torch.tensor(df_data['is_recommended'].values, dtype=torch.float32)
        self.features = all_item_features_tensor

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.features[self.items[idx]], self.labels[idx]
```

### Train-Validation Split
Untuk menguji generalisasi model, data dibagi menjadi data latih (80%) dan validasi (20%) secara stratified agar proporsi label tetap seimbang.
```
from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(df_merged, test_size=0.2, random_state=42, stratify=df_merged['is_recommended'])
train_ds = SteamDatasetWithFeatures(X_train, all_item_features_tensor)
val_ds = SteamDatasetWithFeatures(X_val, all_item_features_tensor)
```

## Modeling and Results

### Arsitektur Model Neural Collaborative Filtering (NCF)
Model yang digunakan dalam proyek ini adalah varian dari Neural Collaborative Filtering (NCF) yang disebut NeuMFWithFeatures, yang menggabungkan pendekatan _Generalized Matrix Factorization (GMF)_ dan _Multi-Layer Perceptron (MLP)_ serta memperhatikan fitur konten game. Model menerima tiga input utama yaitu indeks pengguna, indeks game, dan vektor fitur game (misalnya harga, rating, kompatibilitas platform). Output model berupa probabilitas apakah pengguna akan merekomendasikan game tersebut.

Arsitektur model terdiri dari: 
1. _Embedding Layer_: untuk mewakili user dan item (game) dalam bentuk vektor dense berukuran tetap.
    - user_emb: Tensor ukuran (num_users, dim_embed)
    - item_emb: Tensor ukuran (num_games, dim_embed)

2. _Generalized Matrix Factorization (GMF)_: Melibatkan perkalian elemen demi elemen (element-wise product) antara user dan item embeddings.

3. MLP Path: Menggabungkan embeddings dengan fitur konten item (fitur numerik, kategori, platform), kemudian dilewatkan melalui beberapa layer fully connected.

4. Output Layer: Hasil dari GMF dan MLP digabungkan dan dilewatkan ke output layer dengan aktivasi sigmoid untuk prediksi biner (apakah direkomendasikan atau tidak).
```
import torch.nn as nn

class NeuMFWithFeatures(nn.Module):
    def __init__(self, n_users, n_items, emb_size=32, num_item_features=0):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_size)
        self.item_emb = nn.Embedding(n_items, emb_size)

        self.mlp_input_size = emb_size * 2 + num_item_features
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU())

        self.output_layer_input_size = 32 + emb_size
        self.output = nn.Linear(self.output_layer_input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item, item_features):
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        gmf = user_vec * item_vec
        mlp_input = torch.cat([user_vec, item_vec, item_features], dim=1)
        mlp_output = self.mlp(mlp_input)
        x = torch.cat([gmf, mlp_output], dim=1)
        return self.sigmoid(self.output(x)).squeeze()
```

### Pelatihan Model
Model dilatih menggunakan:
- Loss Function: Binary Cross Entropy (nn.BCELoss) karena target bersifat biner (is_recommended)
- Optimizer: Adam
- Learning rate: 0.001
- Epoch: 3
- Batch Size: 256
```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = NeuMFWithFeatures(n_users_overall, n_items_overall, emb_size=32, num_item_features=num_features_per_item).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
```
> Evaluasi dilakukan di setiap epoch menggunakan metrik klasifikasi yaitu Accuracy, Presisi, Recall, dan F1-Score baik pada data training maupun validasi.

### Top 10 Rekomendasi (Result)
Untuk menyelesaikan permasalahan personalisasi, sistem ini menyediakan output rekomendasi berbentuk Top-N game baru yang diprediksi akan disukai oleh pengguna berdasarkan skor tertinggi dari model.

Fungsi prediksi mengevaluasi seluruh game yang belum pernah dimainkan oleh pengguna, lalu memeringkatnya berdasarkan skor prediksi dari model.
```
# Tampilkan game yang sudah dimainkan pengguna
played_games_info = games[games['app_id'].isin(user_played_games_original_list)][['app_id', 'title', 'rating', 'win', 'mac', 'linux']]
```
Contoh penerapan rekomendasi game yang belum pernah dimainkan pada pengguna `7837098` berdasarkan game-game yang sudah pernah dimainkan:

- Game yang sudah dimainkan oleh pengguna `7837098`:

    | No  | Title                                              | Rating                  | Win  | Mac  | Linux |
    |-----|----------------------------------------------------|-------------------------|------|------|--------|
    | 1   | Ultimate Fishing Simulator                         | Very Positive           | True | False| False  |
    | 2   | Marble It Up!                                      | Very Positive           | True | True | False  |
    | 3   | Five Nights at Freddy's 2                          | Very Positive           | True | False| False  |
    | 4   | Amnesia: A Machine for Pigs                        | Mixed                   | True | True | True   |
    | 5   | I Am Bread                                         | Mostly Positive         | True | True | False  |
    |...|...|...|...|...|...|
    | 21  | Visage                                             | Very Positive           | True | False| False  |
    | 22  | Pogostuck: Rage With Your Friends                  | Very Positive           | True | False| False  |
    | 23  | The Henry Stickmin Collection                      | Overwhelmingly Positive| True | True | False  |
    | 24  | Amnesia: Rebirth                                   | Mostly Positive         | True | False| True   |
    | 25  | Superliminal                                       | Very Positive           | True | True | True   |

- Top 10 Game rekomendasi untuk pengguna `7837098`:

    | No  | Title                                        | Rating                  | Win  | Mac  | Linux | predicted_score |
    |-----|----------------------------------------------|-------------------------|------|------|-------|-----------------|
    | 1   | Cube Escape Collection                        | Overwhelmingly Positive | True | True | False | 0.992425        |
    | 2   | VTOL VR                                      | Overwhelmingly Positive | True | False| False | 0.991820        |
    | 3   | Entropy : Zero 2                             | Overwhelmingly Positive | True | False| False | 0.991582        |
    | 4   | Battle Map Studio                            | Positive                | True | True | False | 0.991194        |
    | 5   | Chacara                                     | Positive                | True | False| True  | 0.990662        |
    | 6   | Of Mice and Moggies                         | Positive                | True | True | True  | 0.990609        |
    | 7   | Zeliria Sanctuary - Rise of Pumpkins        | Positive                | True | True | True  | 0.990422        |
    | 8   | REFLEXIA Prototype ver. Original Graphics    | Positive                | True | False| True  | 0.990254        |
    | 9   | Train Simulator: CSX NRE 3GS-21B 'Genset' Loco... | Positive                | True | False| False | 0.989872        |
    | 10  | Helltaker                                   | Overwhelmingly Positive | True | True | True  | 0.989774        |

Rekomendasi Top 10 game untuk pengguna `7837098` didominasi oleh game dengan rating “`Overwhelmingly Positive`” dan `“Positive”`. Hal ini menunjukkan bahwa model memprioritaskan kualitas dan kepuasan pengguna. Sebagian besar game mendukung Windows, dengan beberapa juga kompatibel di Mac dan Linux, sehingga memperhatikan platform yang digunakan. Skor prediksi yang tinggi menunjukkan keyakinan model bahwa game-game ini sesuai dengan preferensi pengguna.

## Evaluation

### Metrik Evaluasi yang Digunakan:
1. Akurasi (Accuracy): Proporsi interaksi yang terklasifikasi benar `is_recommended` (True (1) atau False (2)).
2. Presisi & Recall & F1-Score: Mengukur kualitas prediksi positif (game yang benar-benar direkomendasikan saat positif), sesuai konteks klasifikasi biner `is_recommended`.

### Hasil Akhir
Evaluasi performa model dilakukan menggunakan metrik klasifikasi biner yaitu accuracy, presisi, recall, dan F1-score. Pemilihan metrik ini disesuaikan dengan tujuan utama dari sistem rekomendasi yang dibangun, yaitu untuk memprediksi apakah pengguna akan merekomendasikan suatu game atau tidak (`is_recommended`). Hasil evaluasi pada pelatihan dan validasi menunjukkan:

> Pelatihan Model

| Epoch | Accuracy | Presisi | Recall | F1-Score | Loss |
|-|-|-|-|-|-|
| 1 | 0.8627 | 0.8662 | 0.9934 | 0.9255 | 0.3708 |
| 2 | 0.8638 | 0.8658 | 0.9956 | 0.9262 | 0.3580 |
| 3 | 0.8682 | 0.8729 | 0.9907 | 0.9281 | 0.3483 |

> Validasi Model

| Epoch | Accuracy | Presisi | Recall | F1-Score |
|-|-|-|-|-|
| 1 | 0.8614 | 0.8657 | 0.9925 | 0.9248 | 
| 2 | 0.8614 | 0.8649 | 0.9938 | 0.9249 |
| 3 | 0.8599 | 0.8686 | 0.9860 | 0.9236 |

 Dari set pelatihan dan validasi terlihat bahwa nilai presisi, recall, dan F1-Score sangat tinggi sehingga menunjukkan bahwa model mampu memberikan prediksi yang relevan terhadap rekomendasi pengguna dan mampu menangkap mayoritas game yang benar-benar disukai pengguna. Selain itu, nilai akurasi pada sudah cukup baik dan stabil serta hanya berbeda sedikit saja antara training dan validasi sehingga menunjukkan bahwa model tidak mengalami overfitting meski hanya dilatih pada tiga epoch.

 Dengan demikian, model ini menunjukkan hasil yang menjanjikan sebagai sistem rekomendasi jika dilihat dari status biner, apakah pengguna merekomendasikan suatu game atau tidak.

### Kesimpulan
Sistem rekomendasi game yang dikembangkan dalam proyek ini berhasil menjawab permasalahan dalam merancang prediksi yang relevan dengan preferensi pengguna dengan memanfaatkan keterkaitan antara data eksplisit interaksi pengguna dan fitur konten game (metadata). 

Model dilatih untuk memprediksi apakah seorang pengguna akan merekomendasikan suatu game menggunakan kolom `is_recommended` dari `recommendations.csv` sebagai variabel target (y). Dengan pendekatan ini, model tidak hanya memetakan interaksi tetapi juga belajar dari kepuasan pengguna. Model Neural Collaborative Filtering yang digunakan mampu merepresentasikan pengguna dan game dalam satu ruang vektor serta menangkap interaksi non-linear serta menghasilkan performa klasifikasi yang baik dengan tingkat akurasi sebesar 85% dan menunjukkan rekomendasi Top-10 yang relevan dalam pengujian. Evaluasi sistem dengan metrik klasifikasi (Akurasi, Presisi, Recall, dan F1-score) membuktikan bahwa model mampu merekomendasikan konten yang sesuai dengan pengalaman pengguna. Dengan pencapaian ini, seluruh goals pengembangan dan evaluasi model telah tercapai. 

Secara strategis, sistem ini berpotensi memberikan dampak signifikan terhadap bisnis, termasuk meningkatkan retensi pengguna, mendorong monetisasi melalui cross-selling, serta membantu pengembang game dalam mempromosikan produknya ke audiens yang lebih tepat sasaran.
