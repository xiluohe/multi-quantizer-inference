# UNCOMMENT TO ADD PATH TO SAVE/CACHE TRANSFORMER MODELS
# import os
# os.environ['TRANSFORMERS_CACHE'] = "/jagupard28/scr0/xiluo-speech/multi-quantizer-experiments/hf/checkpoints/"

import torch
import torchaudio
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from transformers import AutoModel

from lhotse.recipes import download_librispeech, prepare_librispeech
from lhotse import CutSet
from lhotse.dataset import BucketingSampler
from lhotse.dataset.input_strategies import AudioSamples
from torch.utils.data import DataLoader

from tqdm import tqdm
from multi_quantization import QuantizerTrainer

class AudioSamplesDataset(torch.utils.data.Dataset):
    def __init__(self):
      self.collator = AudioSamples()

    def __getitem__(self, cuts: CutSet) -> dict:
        audio_padded, audio_lengths = self.collator(cuts)
        return { "audio_padded": audio_padded, "audio_lengths": audio_lengths }

def load_model(model_path):
    hf_model = AutoModel.from_pretrained(model_path).to('cuda')
    assert hf_model.__class__.__name__ in {"Wav2Vec2Model", "HubertModel"}
    model = import_huggingface_model(hf_model).eval().to('cuda')
    return model

def load_dataset(corpus_dir, download_part=None):
    if download_part:
        download_librispeech(dataset_parts=download_part)
    libri = prepare_librispeech(corpus_dir="LibriSpeech", output_dir="data/")
    return libri

def get_quantizer(model, train_loader, num_codebooks, model_extract_layer=24):
    LAYER_OF_INTEREST=model_extract_layer
    layer_index=LAYER_OF_INTEREST - 1
    
    activations = []
    
    # Register hook to trigger whenever forward() of nth layer is called
    model.encoder.transformer.layers[layer_index].register_forward_hook(
        # Append outputs to list
        lambda model, inputs, outputs: activations.append(outputs.detach())
      )

    batch = next(iter(train_loader))    
    
    final_activations, feat_lens=model(batch['audio_padded'].to('cuda'), batch["audio_lengths"].to('cuda'))
    int_activations=activations[0]
    activations.clear()
    
    trainer = QuantizerTrainer(
        dim=int_activations.shape[-1], bytes_per_frame=num_codebooks, device=torch.device("cuda") 
    )

    return trainer

def train(model, quantizer_trainer, train_loader, model_extract_layer=24, pbar=None): 
    LAYER_OF_INTEREST=model_extract_layer
    layer_index=LAYER_OF_INTEREST - 1
    
    activations = []
    
    # Register hook to trigger whenever forward() of nth layer is called
    model.encoder.transformer.layers[layer_index].register_forward_hook(
        # Append outputs to list
        lambda model, inputs, outputs: activations.append(outputs.detach())
      )
    
    while not quantizer_trainer.done():
    
        batch=next(iter(train_loader))
    
        with torch.no_grad():
            final_activations, feat_lens=model(batch['audio_padded'].to('cuda'), batch["audio_lengths"].to('cuda'))
    
        # Subset 0th item from list and clear list
        int_activations=activations[0]
        activations.clear()
    
        # Retrieve only non-pad frames
        non_pad_activations = []
    
        for item, final_frame in zip(int_activations, feat_lens):
            non_pad_activations.append(item[:final_frame])
    
        # Stack non-pad frames (1 frame=1 item for quantizer training)
        quantizer_train_batch = torch.cat(non_pad_activations, dim=0)
    
        quantizer_trainer.step(quantizer_train_batch)
        if pbar: pbar.update(1)
    
    print("Done!")
    if pbar: pbar.close()

def run(model='facebook/hubert-large-ls960-ft', num_codebooks=16, quantizer_model_save_path='./quantizer.pt', model_extract_layer=24):
    model = load_model(model)
    dset = load_dataset("LibriSpeech", download_part="mini_librispeech")
    
    cuts_train = CutSet.from_manifests(**dset["train-clean-5"])

    train_sampler = BucketingSampler(
        cuts_train,
        max_duration=60,
        shuffle=True,
        drop_last=True
    )
    
    train_loader = DataLoader(
        AudioSamplesDataset(),
        sampler=train_sampler,
        batch_size=None,
        num_workers=1
    )

    quantizer_trainer = get_quantizer(model, train_loader, num_codebooks, model_extract_layer)

    pbar = tqdm(total=len(cuts_train))

    quantizer_trainer = train(model, quantizer_trainer, train_loader, model_extract_layer, pbar)

    quantizer = quantizer_trainer.get_quantizer()
    torch.save(quantizer.state_dict(), quantizer_model_save_path)

if __name__ == "__main__":
    run(model='facebook/hubert-large-ls960-ft', 
        num_codebooks=16, 
        quantizer_model_save_path='./quantizers/full-layer24-16N-quantizer.pt', 
        model_extract_layer=24)
    

    
    
    