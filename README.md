# Bank Customer Churn Prediction (Banka Müşteri Terk Tahmini)

Canlı Demo: [Buraya Streamlit Linkini Ekleyebilirsin]

Bu repo, veri bilimi ve finansal veri analitiği alanındaki portföyüm için geliştirdiğim uçtan uca bir makine öğrenmesi projesidir. Projenin temel amacı, banka müşterilerinin geçmiş demografik ve finansal hareketlerini analiz ederek bankayı terk etme (churn) risklerini önceden tespit etmektir.

## İş Problemi ve Yaklaşım
Sektörde yeni bir müşteri kazanmanın maliyeti, mevcut müşteriyi elde tutmaktan çok daha yüksektir. Bu projede, pazarlama bütçesinin sadece gerçekten bankayı terk etme riski taşıyan müşterilere (elde tutma kampanyalarıyla) harcanabilmesi için bir karar destek modeli geliştirdim.

Modelin performansını değerlendirirken "Accuracy" (Doğruluk) yerine "Recall" (Duyarlılık) metriğine odaklandım. Çünkü iş zekası perspektifinden bakıldığında; kalacak birine yanlışlıkla kampanya sunmak, gerçekten gidecek bir müşteriyi gözden kaçırıp tamamen kaybetmekten daha az maliyetlidir.

## Veri Seti
Kaggle'dan alınan 10.000 satırlık bir veri seti kullanıldı. Veride müşterilerin yaşı, hesap bakiyesi, kredi skoru, ülkesi ve kullandığı ürün sayısı gibi özellikler bulunuyor. Veri setindeki müşterilerin yaklaşık %80'i kalıcı, %20'si ise terk eden (Exited=1) grupta olduğu için dengesiz (imbalanced) bir veri yapısı mevcuttur.

## Proje Adımları ve Optimizasyon
1. Keşifçi Veri Analizi ve Temizlik: Analize katkı sağlamayacak ID ve soyisim gibi değişkenler model dışı bırakıldı. Değişkenlerin hedef üzerindeki etkileri istatistiksel olarak incelendi.
2. Özellik Mühendisliği (Feature Engineering): Veri sızıntısını (data leakage) engellemek adına tüm veri dönüştürme işlemleri (One-Hot Encoding, StandardScaler) Scikit-Learn Pipeline yapısı içinde kurgulandı.
3. Model Eğitimi: Sınıflandırma algoritması olarak Random Forest tercih edildi. Sınıf dengesizliği class_weight="balanced" parametresi ile giderildi.
4. Optuna ile Bayesian Optimizasyonu: Modelin hiperparametrelerini (ağaç sayısı, maksimum derinlik vb.) manuel denemek yerine, Optuna kütüphanesi kullanılarak Recall değerini en üst düzeye çıkaran parametre kombinasyonu yapay zeka ile optimize edildi.
5. Canlıya Alma (Deployment): Optimize edilen model .joblib formatında dışa aktarıldı ve Streamlit kullanılarak son kullanıcının risk hesaplaması yapabileceği interaktif bir web arayüzüne dönüştürüldü.

## Öne Çıkan İçgörüler
Eğitilen modelin değişken önem düzeyleri (Feature Importance) incelendiğinde banka için şu sonuçlar ortaya çıkmıştır:
- Yaş: Bankayı terk etme eğilimi 40-60 yaş bandındaki müşterilerde belirgin bir şekilde daha yüksektir. Modelin en çok ağırlık verdiği değişken yaştır.
- Ürün Sayısı: Müşterinin bankadan kullandığı ürün sayısı 3 veya 4'e ulaştığında ayrılma riski dramatik şekilde artmaktadır. Bu durum, hizmet karmaşıklığından duyulan memnuniyetsizliğe işaret edebilir.
- Aktiflik: Hesaplarında aktif işlem yapmayan müşteriler doğal olarak en yüksek risk grubundadır.

## Kurulum ve Çalıştırma
Projeyi kendi yerel ortamınızda test etmek için aşağıdaki adımları izleyebilirsiniz.

Gerekli kütüphaneleri yükleyin:
pip install -r requirements.txt

Modeli eğitin ve Optuna optimizasyonunu başlatın:
python train.py

Web arayüzünü başlatın:
streamlit run app.py

## İletişim
Mert Gani Gürpınar
Mimar Sinan Güzel Sanatlar Üniversitesi - İstatistik (4. Sınıf)
LinkedIn: www.linkedin.com/in/mert-gani-gurpina
Email: ganigurpinar@gmail.com