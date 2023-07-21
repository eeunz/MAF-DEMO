# MAF-DEMO

![image](https://github.com/eeunz/MAF-DEMO/assets/110804596/fc525522-7432-4d2c-8ab3-88f6f3a371e3)

MAF-DEMO shows how to build a MAF website.
## What it is
Demonstration of possible implementation of web presence for learning purposes. It can be used as a starting point for your own project.
## What's inside
It is not a ready web presence for live operation. Further steps are necessary for achieving this.
## What's inside?
MAF-DEMO currently contains three tablu data and one image data. It also includes 14 algorithms and will continue to supplement them in the future.
* Data : COMPAS, German credit scoring, Adult census income, Public Figures Face Database(Image)
* Algorithm : Disparate_Impact_Remover, Learning_Fair_Representation, Reweighing, Gerry_Fair_Classifier, Meta_Fair_Classifier, Prejudice_Remover, FairBatch, FairFeatureDistillation(Image only), FairnessVAE(Image only), KernelDensityEstimator, LearningFromFairness(Image only)

## How to use
### 1. Data selection
![image](https://github.com/eeunz/MAF-DEMO/assets/110804596/2385e86d-68ff-4fbb-9060-6c0514aacc9d)

샘플 데이터 선택 화면입니다. 현재 Sample 디렉토리에 적합한 파일이 있어야 제대로 실행되며, 데이터는 Preset sample 4가지, Custom dataset 1가지 선택 가능합니다.
* Custom dataset 선택 시 제한사항
  * csv 파일만 가능
   Target, Bias 열이 반드시 하나씩 있어야 하며, 이름도 동일해야함

### 2. Metric
![image](https://github.com/eeunz/MAF-DEMO/assets/110804596/0d07f526-f571-4fe9-b55b-06ad6dcec7d2)
* Data 자체 Bias measures와 Base model (SVM) bias measures, T-SNE analysis를 차트로 표현합니다.

### 3. algorithm select
![image](https://github.com/eeunz/MAF-DEMO/assets/110804596/c6b846d0-106c-41b9-9bf2-af881095ac8c)

편향성 완화 알고리즘 선택 화면입니다. 현재 AIF360의 알고리즘과 컨소시엄에서 개발한 알고리즘을 포함하고 있으며, 향후 추가할 예정입니다. SOTA 알고리즘 중 일부는 Image data로만 활용 가능하며, 현재 Image data는 Pubfig 데이터만 존재합니다. Pubfig 가 아닐 경우 해당 알고리즘들은 disabled 됩니다.



### 4. compare models
![image](https://github.com/eeunz/MAF-DEMO/assets/110804596/c24ffed4-72f0-43d8-a985-4e455af45c2c)
* base model 과 mitigated model 간의 결과를 비교합니다.
* 편향성이 낮은 수치는 하이라이트로 표시됩니다.

