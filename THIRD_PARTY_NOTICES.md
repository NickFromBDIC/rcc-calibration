# Third-party components

The code uses [MONAI](https://github.com/Project-MONAI/MONAI),
[PyRadiomics](https://github.com/AIM-Harvard/pyradiomics),
[MedicalNet](https://github.com/Tencent/MedicalNet), and
[MedVAE](https://github.com/StanfordMIMI/MedVAE).

MedVAE configuration files are derived from Stanford MIMI's public model
configuration. The MedVAE adapter retains the upstream architecture and forward
methods. MedicalNet and MedVAE checkpoint bundles include parameters initialized
from the respective public pretrained models. Their MIT notices are included in
[licenses/medvae.txt](licenses/medvae.txt) and
[licenses/medicalnet.txt](licenses/medicalnet.txt).

[KiTS23](https://github.com/neheller/kits23#license-and-attribution) provides the
CT images, annotations and clinical metadata. Its data terms are CC BY-NC-SA;
the repository's MIT code license does not replace the dataset terms. The
dataset is downloaded from its official sources and is not bundled here.
