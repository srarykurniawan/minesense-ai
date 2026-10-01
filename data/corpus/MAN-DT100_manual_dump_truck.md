---
doc_id: MAN-DT100
title: Manual Perawatan dan Troubleshooting Dump Truck DT-100
doc_type: manual
equipment: DT-100
department: Maintenance
revision: 2
---
# MAN-DT100 Manual Perawatan & Troubleshooting Dump Truck DT-100 (kelas 100 ton)

> Dokumen SINTETIS. "DT-100" adalah model fiktif. Gunakan manual resmi pabrikan (OEM) untuk unit Anda.

## 1. Spesifikasi Ringkas
- Kapasitas muatan nominal: 100 ton. Mesin diesel 12 silinder, 1.050 HP.
- Kapasitas oli mesin: 150 L. Oli hidrolik: 420 L. Coolant: 380 L.
- Tekanan ban (dingin): 7,0 bar.

## 2. Interval Perawatan Berkala (Periodic Maintenance / PM)
| Interval (hour meter) | Pekerjaan utama |
|---|---|
| Setiap 10 jam / harian | P2H, cek level cairan, grease point utama |
| **250 jam** | Ganti oli mesin & filter oli, cek fan belt, sampel oli (SOS) |
| **500 jam** | PM 250 + ganti filter bahan bakar, filter udara luar dibersihkan |
| **1000 jam** | PM 500 + ganti filter oli transmisi & filter hidrolik, cek tekanan suspensi |
| **2000 jam** | PM 1000 + ganti oli hidrolik, oli transmisi, oli final drive, kalibrasi rem |

## 3. Kode Gangguan (Fault Code) dan Troubleshooting
### E-ENG-101 — Temperatur coolant tinggi (> 105 °C)
Kemungkinan penyebab: (1) sirip radiator tersumbat debu/lumpur, (2) level coolant kurang/bocor, (3) fan belt slip atau putus, (4) thermostat macet tertutup, (5) water pump rusak.
Langkah: hentikan unit di lokasi aman dan biarkan mesin idle 5 menit (jangan langsung dimatikan), cek level coolant setelah suhu < 50 °C, bersihkan radiator dengan udara bertekanan/air bertekanan rendah, cek ketegangan fan belt, uji thermostat.

### E-ENG-115 — Tekanan oli mesin rendah
Ambang: < 1,0 bar saat idle atau < 2,5 bar pada rpm tinggi. Kemungkinan penyebab: level oli rendah, filter oli tersumbat, pengenceran oli oleh bahan bakar, pompa oli aus, sensor tekanan rusak.
Langkah: **matikan mesin segera**, cek level oli, ambil sampel oli untuk uji fuel dilution, verifikasi dengan manometer mekanik sebelum mengganti sensor.

### E-TRM-201 — Temperatur oli transmisi tinggi (> 120 °C)
Kemungkinan penyebab: level oli transmisi kurang, oil cooler tersumbat, kopling slip, operator sering menahan unit di gigi rendah dengan rpm tinggi.
Langkah: kurangi beban, cek level oli transmisi, bersihkan cooler, lakukan tes stall untuk deteksi slip kopling.

### E-BRK-301 — Tekanan sistem rem rendah
Kemungkinan penyebab: kebocoran selang/sambungan rem, tekanan nitrogen accumulator rendah, pompa rem aus.
Langkah: **unit dilarang beroperasi (critical item)**, pasang tag DO NOT OPERATE, cek kebocoran, ukur tekanan pengisian nitrogen accumulator, lakukan tes rem statis setelah perbaikan.

### E-SUS-401 — Tekanan silinder suspensi tidak seimbang
Kemungkinan penyebab: kebocoran gas nitrogen atau oli pada silinder suspensi, seal rusak. Gejala: unit miring, guncangan berlebih, frame retak jika dibiarkan.
Langkah: ukur tinggi silinder suspensi (ride height), isi ulang nitrogen sesuai spesifikasi, ganti seal bila bocor.

### E-ELC-501 — Tegangan sistem rendah (< 24 V)
Kemungkinan penyebab: alternator rusak, belt alternator slip, terminal baterai korosi/longgar, kabel ground rusak.
Langkah: ukur tegangan pengisian (normal 27–28,5 V saat mesin hidup), bersihkan terminal, cek output alternator.

### E-HYD-601 — Vessel (hoist) lambat naik
Kemungkinan penyebab: level oli hidrolik rendah, pompa hoist aus, relief valve setting turun, filter suction tersumbat.
Langkah: cek level oli hidrolik, ukur tekanan relief hoist, cek waktu angkat (normal ± 20 detik penuh).

## 4. Catatan Keselamatan Perawatan
- Selalu terapkan LOTO (SOP-K3-002) dan pasang body prop sebelum bekerja di bawah vessel.
- Jangan membuka tutup radiator saat mesin panas.
- Ban raksasa hanya boleh dilepas oleh teknisi ban terlatih; kempeskan ban sebelum dilepas dari rim.
