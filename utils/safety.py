import torch
from transformers import pipeline


class TextSafetyChecker:
    def __init__(
        self,
        model_id="michellejieli/NSFW_text_classifier",
        device=torch.device("cuda:0"),
        token=None,
        local_files_only=False,
    ):
        device_arg = device.index if getattr(device, "type", None) == "cuda" else -1
        self.model = pipeline(
            "text-classification",
            tokenizer=(model_id, {"model_max_length": 512}),
            model=model_id,
            device=device_arg,
            token=token,
            local_files_only=local_files_only,
        )

    @torch.no_grad()
    def check_text(self, texts):
        return self.model(texts, truncation=True)


class ImageSafetyChecker:
    def __init__(self, model_id="Marqo/nsfw-image-detection-384", device=torch.device("cuda:0")):
        import timm

        self.model = timm.create_model(f"hf_hub:{model_id}", pretrained=True).to(device)
        self.model = self.model.eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.device = device

    @torch.no_grad()
    def is_nsfw(self, image):
        output = self.model(self.transforms(image).unsqueeze(0).to(self.device))
        class_names = self.model.pretrained_cfg["label_names"]
        return class_names[output[0].argmax()] == "NSFW"
