# Günlük Git İş Akışı (Solo Geliştirici)

Bu rehber, tek başına çalışırken projeyi temiz ve düzenli tutmak için basit bir günlük rutindir.

---

## 🌞 1. Güne Başlarken (Durum Kontrolü)

Kod yazmaya başlamadan önce her şeyin doğru yerde olduğundan emin ol.

1.  **Hangi Branch'deyim?**
    * **Komut:** `git branch`
    * **Amaç:** Aktif branch'in `* dev` (veya üzerinde çalıştığınız branch) olduğunu doğrulamak. Eğer `main` ise, `git checkout dev` ile geliştirme branch'ine geç.

2.  **Dünden Kalan Değişiklik Var mı?**
    * **Komut:** `git status`
    * **Amaç:** Ekranda `nothing to commit, working tree clean` (Kaydedilecek bir şey yok, çalışma alanı temiz) mesajını görmek. Eğer kaydedilmemiş dosyalar varsa, önce onları hallet.

---

## 💻 2. Geliştirme

Bu aşamada Git'e ihtiyacın yok.

* Kodunu yaz.
* Test et.
* Çalıştığından emin ol.

---

## 🌝 3. Günü Bitirirken (Kaydet ve Yükle)

Çalışmanı bitirdiğinde veya önemli bir aşamayı tamamladığında, ilerlemeni kaydet ve GitHub'a yedekle.

1.  **Neleri Değiştirdim?**
    * **Komut:** `git status`
    * **Amaç:** Değiştirdiğin veya eklediğin dosyaları gözden geçir. `.gitignore`'a eklemen gereken gereksiz bir dosya (örn: `.log`, `.csv`, `.zip`) var mı diye kontrol et.

2.  **Tüm Değişiklikleri Hazırla**
    * **Komut:** `git add .`
    * **Amaç:** Tüm geçerli değişiklikleri "kayıt paketine" ekler.

3.  **Değişiklikleri Etiketle (Kaydet)**
    * **Komut:** `git commit -m "Yaptığın değişikliği anlatan kısa bir mesaj"`
    * **Örnek:** `git commit -m "Hata ayıklama logları eklendi"`
    * **Amaç:** Pakete bir etiket yapıştırarak ne yaptığını gelecekteki sana hatırlatmak.

4.  **GitHub'a Gönder (Yedekle)**
    * **Komut:** `git push origin dev`
    * **Amaç:** Bilgisayarındaki kayıtlı değişiklikleri GitHub'daki `dev` branch'ine göndermek. Artık kodun güvende!