import glob
import os
from decord import AudioReader, VideoReader, cpu
from latentsync.whisper.audio2feature import Audio2Feature


if __name__ == "__main__":    
    audio_encoder = Audio2Feature(model_path="../LatentSync/checkpoints/whisper/tiny.pt")
    audio_path = "/home/tmpuser/data3/Ekatai/trainset-eikitai3/1869.mp4"
    # array = audio_encoder.audio2feat(audio_path)
    # print(array.shape)
    # fps = 25
    # whisper_idx_multiplier = 50.0 / fps

    # i = 0
    # print(f"video in {fps} FPS, audio idx in 50FPS")
    # while True:
    #     start_idx = int(i * whisper_idx_multiplier)
    #     selected_feature, selected_idx = audio_encoder.get_sliced_feature(feature_array=array, vid_idx=i, fps=fps)
    #     print(f"video idx {i},\t audio idx {selected_idx},\t shape {selected_feature.shape}")
    #     i += 1
    #     if start_idx > len(array):
    #         break