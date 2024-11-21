"""
VEnhancer: A text-guided video enhancement model that can upscale resolution,
adjust frame rates, and enhance video quality based on text prompts.
"""

import os
import glob
from typing import List, Optional
import torch
from easydict import EasyDict
from huggingface_hub import hf_hub_download

from VEnhancer.inference_utils import (
    get_logger,
    load_video,
    preprocess,
    adjust_resolution,
    make_mask_cond,
    collate_fn,
    tensor2vid,
    save_video,
    load_prompt_list,
)
from VEnhancer.video_to_video.utils.seed import setup_seed
from VEnhancer.video_to_video.video_to_video_model import VideoToVideo
from VEnhancer.configs.venhnacer_config import VEnhancerConfig

logger = get_logger()


class VEnhancer:
    """Video enhancement model with text guidance.

    This class implements a video enhancement model that can upscale resolution,
    adjust frame rates, and enhance video quality based on text prompts. It utilizes
    deep learning techniques to process and enhance video content effectively.

    Attributes:
        config (VEnhancerConfig): Configuration settings for the VEnhancer model.
        model (VideoToVideo): The underlying video enhancement model.
        model_path (str): Path to the model checkpoint.
    """

    def __init__(self, config: Optional[VEnhancerConfig] = None):
        """Initialize the VEnhancer model.

        This constructor initializes the VEnhancer model with the provided configuration.
        If no configuration is provided, a default configuration is used. It also checks
        for the existence of the model checkpoint and downloads it if necessary.

        Args:
            config (Optional[VEnhancerConfig]): VEnhancerConfig object containing model settings.
        
        Raises:
            AssertionError: If the model checkpoint does not exist after initialization.
        """
        self.config = config or VEnhancerConfig()
        
        if not self.config.model_path:
            self.download_model()
        else:
            self.model_path = self.config.model_path
            
        assert os.path.exists(self.model_path), "Error: checkpoint Not Found!"
        logger.info(f"checkpoint_path: {self.model_path}")

        os.makedirs(self.config.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__="model_cfg")
        model_cfg.model_path = self.model_path
        self.model = VideoToVideo(model_cfg)

    def enhance_a_video(
        self, 
        video_path: str, 
        prompt: str,
        up_scale: Optional[float] = None,
        target_fps: Optional[int] = None,
        noise_aug: Optional[int] = None,
    ) -> str:
        """Enhance a video using text guidance.

        This method processes the input video based on the provided text prompt and
        enhances its quality by adjusting resolution, frame rate, and applying noise
        augmentation as specified.

        Args:
            video_path (str): Path to the input video file.
            prompt (str): Text prompt for enhancement guidance.
            up_scale (Optional[float]): Optional upscaling factor (overrides config).
            target_fps (Optional[int]): Optional target FPS (overrides config).
            noise_aug (Optional[int]): Optional noise augmentation level (overrides config).

        Returns:
            str: Path to the enhanced video file.

        Raises:
            ValueError: If the noise augmentation level is out of bounds.
        """
        up_scale = up_scale or self.config.up_scale
        target_fps = target_fps or self.config.target_fps
        noise_aug = noise_aug or self.config.noise_aug

        save_name = os.path.splitext(os.path.basename(video_path))[0]
        caption = prompt + self.model.positive_prompt
        logger.info(f"Processing with prompt: {prompt}")

        # Load and preprocess video
        input_frames, input_fps = load_video(video_path)
        in_f_num = len(input_frames)
        logger.info(f"Input frames: {in_f_num}, FPS: {input_fps}")

        # Calculate frame interpolation
        interp_f_num = max(round(target_fps / input_fps) - 1, 0)
        interp_f_num = min(interp_f_num, 8)
        target_fps = input_fps * (interp_f_num + 1)
        logger.info(f"Target FPS: {target_fps}")

        # Process video data
        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape
        target_h, target_w = adjust_resolution(h, w, up_scale)
        logger.info(f"Resolution: {h}x{w} → {target_h}x{target_w}")

        # Prepare conditioning
        mask_cond = torch.Tensor(make_mask_cond(in_f_num, interp_f_num)).long()
        noise_aug = min(max(noise_aug, 0), 300)
        logger.info(f"Noise augmentation: {noise_aug}")

        # Prepare inference data
        pre_data = {
            "video_data": video_data,
            "y": caption,
            "mask_cond": mask_cond,
            "s_cond": self.config.s_cond,
            "interp_f_num": interp_f_num,
            "target_res": (target_h, target_w),
            "t_hint": noise_aug,
        }

        setup_seed(self.config.seed)

        # Run inference
        with torch.no_grad():
            data_tensor = collate_fn(pre_data, "cuda:0")
            output = self.model.test(
                data_tensor,
                total_noise_levels=900,
                steps=self.config.steps,
                solver_mode=self.config.solver_mode,
                guide_scale=self.config.guide_scale,
                noise_aug=noise_aug,
            )

        # Save results
        output = tensor2vid(output)
        save_video(output, self.config.result_dir, f"{save_name}.mp4", fps=target_fps)
        return os.path.join(self.config.result_dir, save_name)

    def download_model(self) -> None:
        """Download model checkpoint from Hugging Face.

        This method downloads the model checkpoint from the Hugging Face repository
        if it does not already exist locally. The filename is determined based on the
        version specified in the configuration.

        Raises:
            RuntimeError: If the model download fails.
        """
        filename = "venhancer_v2.pt" if self.config.version == "v2" else "venhancer_paper.pt"
        ckpt_dir = "./ckpts/"
        os.makedirs(ckpt_dir, exist_ok=True)
        
        local_file = os.path.join(ckpt_dir, filename)
        if not os.path.exists(local_file):
            logger.info("Downloading the VEnhancer checkpoint...")
            try:
                hf_hub_download(
                    repo_id=self.config.repo_id,
                    filename=filename,
                    local_dir=ckpt_dir
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
        self.model_path = local_file

    def process_batch(
        self,
        input_path: str,
        prompt: Optional[str] = None,
        prompt_path: Optional[str] = None,
        filename_as_prompt: bool = False,
    ) -> List[str]:
        """Process multiple videos in batch.

        This method processes a batch of videos, enhancing each based on the provided
        prompts. It can handle both individual video files and directories containing
        multiple videos.

        Args:
            input_path (str): Path to input video or directory.
            prompt (Optional[str]): Optional text prompt for all videos.
            prompt_path (Optional[str]): Optional path to a file containing prompts.
            filename_as_prompt (bool): If True, use the filename as the prompt.

        Returns:
            List[str]: List of paths to the enhanced video files.

        Raises:
            TypeError: If input_path is neither a file nor a directory.
            AssertionError: If the number of prompts does not match the number of videos.
        """
        # Get list of video files
        if os.path.isdir(input_path):
            file_path_list = sorted(glob.glob(os.path.join(input_path, "*.mp4")))
        elif os.path.isfile(input_path):
            file_path_list = [input_path]
        else:
            raise TypeError("input must be a directory or video file!")

        # Handle prompts
        prompt_list = None
        if os.path.isfile(prompt_path or ""):
            prompt_list = load_prompt_list(prompt_path)
            assert len(prompt_list) == len(file_path_list), "Number of prompts must match number of videos."

        enhanced_paths = []
        for idx, file_path in enumerate(file_path_list):
            logger.info(f"Processing video {idx + 1}/{len(file_path_list)}")
            
            # Determine prompt for current video
            current_prompt = prompt
            if filename_as_prompt:
                current_prompt = os.path.splitext(os.path.basename(file_path))[0]
            elif prompt_list is not None:
                current_prompt = prompt_list[idx]
            elif not current_prompt:
                prompt_file = os.path.splitext(file_path)[0] + ".txt"
                if os.path.isfile(prompt_file):
                    current_prompt = load_prompt_list(prompt_file)[0]
                else:
                    current_prompt = "a good video"

            # Process video
            output_path = self.enhance_a_video(file_path, current_prompt)
            enhanced_paths.append(output_path)

        return enhanced_paths
