import os
from pathlib import Path
import torch
import io
import fitz  
import sys
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel, StoppingCriteria, StoppingCriteriaList
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from typing import Optional, List


processor = AutoProcessor.from_pretrained("facebook/nougat-small")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class OCR:
    def __init__(self):
        pass
    
    def rasterize_paper(self, pdf: Path, dpi: int = 96) -> List[io.BytesIO]:
        pillow_images = []
        try:
            if isinstance(pdf, (str, Path)):
                pdf = fitz.open(pdf)
            pages = range(len(pdf))
            for i in pages:
                page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
                pillow_images.append(io.BytesIO(page_bytes))
        except Exception as e:
            print(f"Error rasterizing paper: {e}")
        return pillow_images

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        images = self.rasterize_paper(pdf=pdf_path)
        text = ""
        for img in images:
            image = Image.open(img)

            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

            outputs = model.generate(pixel_values,
                                     min_length=1,
                                     max_length=3584,
                                     bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                     return_dict_in_generate=True,
                                     output_scores=True,
                                     stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
                                     )
            generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
            generated = processor.post_process_generation(generated, fix_markdown=False)
            text += generated + "\n\n"
        return text

class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)

class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0

if __name__ == "__main__":
    # test with file path 
    pdf_path = ""
    ocr = OCR()
    text = ocr.extract_text_from_pdf(pdf_path=pdf_path)
