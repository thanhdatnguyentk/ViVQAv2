# Bao Cao Phan Tich Visual Grounding - ViVQAv2

## 1. Tong Quan Kich Thuoc Du Lieu
- **Tong so cau hoi phan tich:** 50016
- **Tong so thuc the trich xuat duoc:** 92665

## 2. Phan Tich Theo Split

| Split | Cau hoi | QGR | EGR | OCR | Nhan xet |
|---|---|---|---|---|---|
| **Train** | 40011 | **38.90%** | 22.42% | 19.09% | Knowledge-driven |
| **Dev** | 5001 | **38.15%** | 21.92% | 18.48% | Knowledge-driven |
| **Test** | 5004 | **38.35%** | 22.32% | 18.66% | Knowledge-driven |
| ****All**** | 50016 | **38.77%** | 22.36% | 18.99% | Knowledge-driven |

## 3. Cac Chi So Tong Hop (All Splits)

| Chi So | Gia tri | Y nghia |
|---|---|---|
| **Question Grounding Rate (QGR)** | **38.77%** | 19393/50016 cau hoi co it nhat mot thuc the lien ket duoc voi anh. |
| **Entity Grounding Ratio (EGR)** | **22.36%** | 20723/92665 thuc the duoc tim thay trong anh. |
| **Object Coverage Rate (OCR)** | **18.99%** | Trung binh mot cau hoi de cap den 18.99% so vat the phat hien duoc trong anh. |

## 4. Phan Loai Cau Hoi (Question Type Taxonomy)

| Phan Loai | Ty le (%) | So luong | Mo ta |
|---|---|---|---|
| **Fully Grounded** | 10.19% | 5095 | Moi thuc the trong cau hoi deu co trong anh (visual-driven manh). |
| **Partially Grounded** | 28.59% | 14298 | Chi tim thay mot phan thuc the trong anh. |
| **Ungrounded** | 60.03% | 30024 | Co thuc the nhung KHONG tim thay trong anh (knowledge-driven/suy luan). |
| **Abstract/Reasoning** | 1.20% | 599 | Khong trich xuat duoc thuc the nao (cau hoi truu tuong/suy luan logic). |

## 5. Taxonomy Theo Split

### Train (40011 cau hoi)

| Phan Loai | Ty le (%) | So luong |
|---|---|---|
| Fully Grounded     | 10.24% | 4096 |
| Partially Grounded | 28.67% | 11470 |
| Ungrounded         | 59.92% | 23975 |
| Abstract/Reasoning | 1.17% | 470 |
| **QGR**            | **38.90%** | 15566/40011 |

### Dev (5001 cau hoi)

| Phan Loai | Ty le (%) | So luong |
|---|---|---|
| Fully Grounded     | 9.80% | 490 |
| Partially Grounded | 28.35% | 1418 |
| Ungrounded         | 60.43% | 3022 |
| Abstract/Reasoning | 1.42% | 71 |
| **QGR**            | **38.15%** | 1908/5001 |

### Test (5004 cau hoi)

| Phan Loai | Ty le (%) | So luong |
|---|---|---|
| Fully Grounded     | 10.17% | 509 |
| Partially Grounded | 28.18% | 1410 |
| Ungrounded         | 60.49% | 3027 |
| Abstract/Reasoning | 1.16% | 58 |
| **QGR**            | **38.35%** | 1919/5004 |

## Ket Luan
- Voi **QGR = 38.77%**, dataset ViVQAv2 nghieng ve **Knowledge-driven**.
- Ti le cau hoi can suy luan ngoai anh (Ungrounded + Abstract): **61.23%**.
