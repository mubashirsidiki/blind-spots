# Qwen3-VL-2B-Instruct Evaluation on Pakistani Urdu Menus

- **Model**: [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- **Evaluated Images**: 5
- **Device**: NVIDIA GeForce RTX 4060 Laptop GPU
- **Total Runtime**: 176.30s

## Performance & Timing Breakdown

| Operation | Total Time | Percentage of Total Runtime |
|---|---|---|
| Model & Processor Loading | 4.02s | 2.3% |
| Image Preprocessing & Tokenization | 219.2ms | 0.1% |
| Model Generation Forward Pass | 171.89s | 97.5% |
| Output Decoding | 12.7ms | 0.0% |

- **Total Tokens Generated**: 3206
- **Average Generation Speed**: 18.7 tokens/second

## Page: `1.jpg`

### Test: `full_transcription`
- **Prompt**: Read all text from this restaurant cover page, including English and Urdu, exactly as written.
- **Latency**: 4.137s (Gen: 4.05s, Speed: 15.8 tok/s, Tokens: 64)

**Model Output**:
```
Enjoy The Taste Of Our
Exclusive Recipes
NOTE: PRICES ARE INCLUSIVE OF TAXES.
جاری کوئی اور برائے نہیں ہے
https://www.facebook.com/Nooranikabab11?mibextid=ZbWKwL
```

## Page: `2.jpg`

### Test: `english_extraction`
- **Prompt**: Extract all English dish names and their prices from this menu. Format as a clean list: 'Dish Name - Price'.
- **Latency**: 10.35s (Gen: 10.338s, Speed: 24.0 tok/s, Tokens: 248)

**Model Output**:
```
KATA KAT MIX PLATE - 1430
KATA KAT MIX PLATE (WITH BUTTER) - 1630
BRAIN MASALA - 1100
BRAIN MASALA (WITH BUTTER) - 1300
KIDNEY MASALA - 880
KIDNEY MASALA (WITH BUTTER) - 1080
MUTTON CHAMPS MASALA - 880
MUTTON CHAMPS MASALA (WITH BUTTER) - 1080
KAPOORA MASALA - 880
KAPOORA MASALA (WITH BUTTER) - 1080
CHICKEN KATA KAT (WITH BONE) - 1800 / 900
CHICKEN KATA KAT (WITH BUTTER) - 2200 / 1100
CHICKEN BONELESS KATA KAT - 2280 / 1140
CHICKEN BONELESS KATA KAT (WITH BUTTER) - 2680 / 1340
```

### Test: `urdu_extraction`
- **Prompt**: Extract all Urdu dish names and their prices from this menu. Format as a clean list: 'اردو نام - قیمت'.
- **Latency**: 14.25s (Gen: 14.231s, Speed: 22.3 tok/s, Tokens: 318)

**Model Output**:
```
- کاٹ کاٹ میکس پلاٹ - 1430
- کاٹ کاٹ میکس پلاٹ (بٹر) - 1630
- براين ماسلا - 1100
- براين ماسلا (بٹر) - 1300
- کیڈنی ماسلا - 880
- کیڈنی ماسلا (بٹر) - 1080
- موتون چامپس ماسلا - 880
- موتون چامپس ماسلا (بٹر) - 1080
- کاپورا ماسلا - 880
- کاپورا ماسلا (بٹر) - 1080
- چیکن کاٹ کاٹ (بٹر) - 1800 / 900
- چیکن کاٹ کاٹ (بٹر) - 2200 / 1100
- چیکن بونلیس کاٹ کاٹ - 2280 / 1140
- چیکن بونلیس کاٹ کاٹ (بٹر) - 2680 / 1340
```

### Test: `bilingual_alignment`
- **Prompt**: For each item in this menu, extract the English dish name, the Urdu dish name, and the price in PKR. Format each item on a new line.
- **Latency**: 7.381s (Gen: 7.37s, Speed: 19.1 tok/s, Tokens: 141)

**Model Output**:
```
KATA KAT MIX PLATE, KATA KAT MIX PLATE (WITH BUTTER), BRAIN MASALA, BRAIN MASALA (WITH BUTTER), KIDNEY MASALA, KIDNEY MASALA (WITH BUTTER), MUTTON CHAMPS MASALA, MUTTON CHAMPS MASALA (WITH BUTTER), KAPOORA MASALA, KAPOORA MASALA (WITH BUTTER), CHICKEN, CHICKEN KATA KAT (WITH BONE), CHICKEN KATA KAT (WITH BUTTER), CHICKEN BONELESS KATA KAT, CHICKEN BONELESS KATA KAT (WITH BUTTER)
```

## Page: `3.jpg`

### Test: `english_extraction`
- **Prompt**: Extract all English dish names and their prices from this menu. Format as a clean list: 'Dish Name - Price'.
- **Latency**: 18.893s (Gen: 18.883s, Speed: 19.2 tok/s, Tokens: 363)

**Model Output**:
```
- BRAIN (PER PIECE) - 550
- KIDNEY (PER PIECE) - 220
- CHAMP (PER PIECE) - 220
- DIL (PER PICE) - 220
- KAPOORA (PER PIECE) - 220
- CHICKEN (PER PIECE) - 450
- CHICKEN BONELESS (PER PIECE) - 570
- BUTTER (PER PIECE) - 100
- PESHAWARI KARAHI - 3800 / 2000
- PESHAWARI KARAHI (WITH BUTTER) - 4200 / 2200
- PESHAWARI WHITE KARAHI - 3900 / 2100
- PESHAWARI WHITE KARAHI (WITH BUTTER) - 4300 / 2300
- PESHAWARI BROWN KARAHI - 3800 / 2000
- PESHAWARI KARAHI - 2200 / 1200
- PESHAWARI KARAHI (WITH BUTTER) - 2600 / 1400
- PESHAWARI WHITE KARAHI - 2300 / 1300
- PESHAWARI WHITE KARAHI (WITH BUTTER) - 2700 / 1500
- BROWN KARAHI - 2200 / 1200
```

### Test: `urdu_extraction`
- **Prompt**: Extract all Urdu dish names and their prices from this menu. Format as a clean list: 'اردو نام - قیمت'.
- **Latency**: 20.273s (Gen: 20.263s, Speed: 18.5 tok/s, Tokens: 374)

**Model Output**:
```
- براين - 550
- کیندی - 220
- شامپ - 220
- دیل - 220
- کاپورا - 220
- چینکن - 450
- چینکن بونل - 570
- باتر - 100
- مUTTON - 3800 / 2000
- پESHAWARI کاراھی (بیت) - 4200 / 2200
- پESHAWARI کاراھی (بیت) - 3900 / 2100
- پESHAWARI ہیلی کاراھی (بیت) - 4300 / 2300
- پESHAWARI بروں کاراھی - 3800 / 2000
- چینکن - 2200 / 1200
- پESHAWARI کاراھی - 2200 / 1200
- پESHAWARI کاراھی (بیت) - 2600 / 1400
- پESHAWARI کاراھی (بیت) - 2300 / 1300
- پESHAWARI ہیلی کاراھی (بیت) - 2700 / 1500
- بروں کاراھی - 2200 / 1200
```

### Test: `bilingual_alignment`
- **Prompt**: For each item in this menu, extract the English dish name, the Urdu dish name, and the price in PKR. Format each item on a new line.
- **Latency**: 27.844s (Gen: 27.833s, Speed: 19.9 tok/s, Tokens: 553)

**Model Output**:
```
- BRAIN (PER PIECE) 550 (پیسہ)
- KIDNEY (PER PIECE) 220 (کیڈی)
- CHAMP (PER PIECE) 220 (چمپ)
- DIL (PER PICE) 220 (دل)
- KAPOORA (PER PIECE) 220 (کاپور)
- CHICKEN (PER PIECE) 450 (چکن)
- CHICKEN BONELESS (PER PIECE) 570 (چکن)
- BUTTER (PER PIECE) 100 (بٹر)

- MUTTON
  - PESHAWARI KARAHI 3800 / 2000 (پیشواوی کاراہی)
  - PESHAWARI KARAHI (WITH BUTTER) 4200 / 2200 (پیشواوی کاراہی (بٹر))
  - PESHAWARI WHITE KARAHI 3900 / 2100 (پیشواوی ہیلی کاراہی)
  - PESHAWARI WHITE KARAHI (WITH BUTTER) 4300 / 2300 (پیشواوی ہیلی کاراہی (بٹر))
  - PESHAWARI BROWN KARAHI 3800 / 2000 (پیشواوی براون کاراہی)

- CHICKEN
  - PESHAWARI KARAHI 2200 / 1200 (پیشواوی کاراہی)
  - PESHAWARI KARAHI (WITH BUTTER) 2600 / 1400 (پیشواوی کاراہی (بٹر))
  - PESHAWARI WHITE KARAHI 2300 / 1300 (پیشواوی ہیلی کاراہی)
  - PESHAWARI WHITE KARAHI (WITH BUTTER) 2700 / 1500 (پیشواوی ہیلی کاراہی (بٹر))
  - BROWN KARAHI 2200 / 1200 (برون کاراہی)
```

## Page: `4.jpg`

### Test: `english_extraction`
- **Prompt**: Extract all English dish names and their prices from this menu. Format as a clean list: 'Dish Name - Price'.
- **Latency**: 9.381s (Gen: 9.37s, Speed: 17.8 tok/s, Tokens: 167)

**Model Output**:
```
- Noorani Special Kabab Beef - 640
- Behari Boti Beef - 680
- Seekh Boti Beef - 590
- Afghan Boti Beef - 700
- Chicken Tikka - 430
- Chicken Malai Tikka - 450
- Bihari Chicken Tikka - 450
- Chicken Boti - 860
- Chicken Malai Boti - 790
- Chicken Bihari Boti Boneless - 800
- Chicken Reshmi Kebab - 540
- Mutton Champ - 900
- Mutton Kapoora - 880
- Mutton Kidney - 880
```

### Test: `urdu_extraction`
- **Prompt**: Extract all Urdu dish names and their prices from this menu. Format as a clean list: 'اردو نام - قیمت'.
- **Latency**: 14.919s (Gen: 14.908s, Speed: 16.3 tok/s, Tokens: 243)

**Model Output**:
```
- نورانی سپیشل کاباب بیف - 640
- بہہری بیت بیف - 680
- سکھ بیت بیف - 590
- افغانی بیت بیف - 700
- چینکی تکا - 430
- چینکی مالای تکا - 450
- بیہاری چینکی تکا - 450
- چینکی بیت - 860
- چینکی مالای بیت - 790
- چینکی بیہاری بیت بونلیس - 800
- چینکی رشمنی کیب - 540
- مUTTON چمپ - 900
- مUTTON کاپورا - 880
- مUTTON کینی - 880
```

### Test: `bilingual_alignment`
- **Prompt**: For each item in this menu, extract the English dish name, the Urdu dish name, and the price in PKR. Format each item on a new line.
- **Latency**: 22.142s (Gen: 22.132s, Speed: 16.3 tok/s, Tokens: 360)

**Model Output**:
```
- NOORANI SPECIAL KABAB BEEF, نورانی سپیشل کاباب بیف, 640
- BEHARI BOTI BEEF, بہاری بٹی بیف, 680
- SEEKH BOTI BEEF, سکھ بٹی بیف, 590
- AFGHANI BOTI BEEF, افغانی بٹی بیف, 700
- CHICKEN TIKKA, چیکن تکا, 430
- CHICKEN MALAI TIKKA, چیکن مالائی تکا, 450
- BIHARI CHICKEN TIKKA, بیہاری چیکن تکا, 450
- CHICKEN BOTI, چیکن بٹی, 860
- CHICKEN MALAI BOTI, چیکن مالائی بٹی, 790
- CHICKEN BIHARI BOTI BONELESS, چیکن بیہاری بٹی بونلیس, 800
- CHICKEN RESHMI KEBAB, چیکن ریشمی کباب, 540
- MUTTON CHAMP, مUTTON چامپ, 900
- MUTTON KAPOORA, مUTTON کاپورا, 880
- MUTTON KIDNEY, مUTTON کینی, 880
```

## Page: `5.jpg`

### Test: `english_extraction`
- **Prompt**: Extract all English dish names and their prices from this menu. Format as a clean list: 'Dish Name - Price'.
- **Latency**: 6.889s (Gen: 6.874s, Speed: 16.6 tok/s, Tokens: 114)

**Model Output**:
```
- Green Salad - 120
- Raita - 150
- Nan - 30
- Masala Nan - 50
- Chapati - 20
- Paratha - 120
- Pepsi Can - 130
- Marinda Can - 130
- 7UP - Can - 130
- Diet 7UP Can - 130
- Mineral Water Large - 130
- Mineral Water Small - 70
```

### Test: `urdu_extraction`
- **Prompt**: Extract all Urdu dish names and their prices from this menu. Format as a clean list: 'اردو نام - قیمت'.
- **Latency**: 9.812s (Gen: 9.799s, Speed: 16.5 tok/s, Tokens: 162)

**Model Output**:
```
- گرین سالاد - 120
- رائٹا - 150
- نان - 30
- ماسلا نان - 50
- شاپاتی - 20
- پاراثا - 120
- پیپسی کان - 130
- مارینا کان - 130
- 7UP کان - 130
- دیت 7UP کان - 130
- مینرل ویت گریم لارج - 130
- مینرل ویت گریم سمال - 70
```

### Test: `bilingual_alignment`
- **Prompt**: For each item in this menu, extract the English dish name, the Urdu dish name, and the price in PKR. Format each item on a new line.
- **Latency**: 5.85s (Gen: 5.837s, Speed: 17.0 tok/s, Tokens: 99)

**Model Output**:
```
GREEN SALAD 120
RAITA 150
NAN 30
MASALA NAN 50
CHAPATI 20
PARATHA 120
PEPSI CAN 130
MARINDA CAN 130
7UP - CAN 130
DIET 7UP CAN 130
MINERAL WATER LARGE 130
MINERAL WATER SMALL 70
```

