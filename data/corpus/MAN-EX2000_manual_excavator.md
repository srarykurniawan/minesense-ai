---
doc_id: MAN-EX2000
title: Manual Perawatan dan Troubleshooting Excavator EX-2000
doc_type: manual
equipment: EX-2000
department: Maintenance
revision: 2
---
# MAN-EX2000 Manual Perawatan & Troubleshooting Excavator EX-2000 (kelas 200 ton)

> Dokumen SINTETIS. "EX-2000" adalah model fiktif. Gunakan manual resmi pabrikan untuk unit Anda.

## 1. Spesifikasi Ringkas
- Bobot operasi ± 200 ton, kapasitas bucket 12 m³ (backhoe).
- Oli hidrolik: 1.200 L. Suhu kerja normal oli hidrolik: 50–80 °C.
- Tekanan pilot normal: 30–35 bar. Tekanan relief utama: 310 bar.

## 2. Interval Perawatan
| Interval | Pekerjaan utama |
|---|---|
| Harian | P2H, greasing pin boom-arm-bucket (auto lube cek level), cek level oli hidrolik |
| **250 jam** | Ganti oli mesin & filter, cek oli swing machinery |
| **500 jam** | Ganti filter pilot, filter bahan bakar, cek track tension |
| **1000 jam** | Ganti oli swing machinery, filter return hidrolik |
| **4000 jam** | Ganti oli hidrolik (atau berdasarkan hasil analisis oli) |

## 3. Kode Gangguan dan Troubleshooting
### X-HYD-110 — Temperatur oli hidrolik tinggi (> 95 °C)
Kemungkinan penyebab: oil cooler tersumbat debu/batubara, level oli hidrolik rendah, relief valve bocor (internal leakage) sehingga oli bersirkulasi pada tekanan tinggi, fan cooler tidak berputar optimal.
Langkah: kurangi beban kerja, cek level oli, bersihkan oil cooler, ukur kecepatan fan, cek temperatur di sisi relief valve dengan thermo-gun untuk mendeteksi kebocoran internal.

### X-HYD-120 — Tekanan pilot rendah (< 30 bar)
Gejala: seluruh gerakan lambat atau tidak responsif. Kemungkinan penyebab: filter pilot tersumbat, pompa pilot aus, relief pilot valve rusak.
Langkah: ganti filter pilot, ukur tekanan pilot pada test port, setel ulang relief pilot.

### X-SWG-210 — Swing lambat atau berbunyi abnormal
Kemungkinan penyebab: oli swing machinery kurang atau terkontaminasi serpihan logam, gear swing aus, swing brake tidak release sempurna, swing motor bocor internal.
Langkah: cek level dan kondisi oli swing machinery (magnetic plug), uji drain rate swing motor, periksa swing brake release pressure.

### X-ENG-310 — Mesin overheat
Kemungkinan penyebab: radiator tersumbat, coolant kurang, fan belt, thermostat. Langkah serupa dengan prosedur overheat pada dump truck: idle-kan mesin sebelum mematikan, bersihkan radiator.

### X-UND-410 — Track kendur atau lepas
Kemungkinan penyebab: tekanan grease track adjuster turun (seal bocor), idler/roller aus.
Langkah: ukur sag track (normal 30–50 mm), tambah grease pada adjuster, periksa kebocoran seal adjuster.

### X-ELC-510 — Error komunikasi kontroler (CAN bus)
Kemungkinan penyebab: konektor longgar/korosi, kabel harness terkelupas akibat gesekan, resistor terminasi rusak.
Langkah: ukur resistansi CAN (normal ± 60 ohm antara CAN-H dan CAN-L), periksa konektor di area boom foot.

## 4. Catatan Keselamatan
- Turunkan bucket ke tanah sebelum perawatan; terapkan LOTO.
- Gunakan akses tangga tiga titik kontak dan harness saat bekerja di atas deck (≥ 1,8 m).
- Tekanan hidrolik sisa harus dibuang sebelum membuka sambungan selang.
