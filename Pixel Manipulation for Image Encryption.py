import numpy as np
from PIL import Image
import os
from typing import Tuple
import random
import time
from datetime import datetime

class PixelEncryption:
    def __init__(self):
        """Initialize the encryption tool with transformation matrices"""
        self.transformation_key = None
        self.block_size = 8
        
    def generate_key(self) -> dict:
        """Generate a complex encryption key with multiple components"""
        random.seed(int(time.time()))
        return {
            'pixel_shift': random.randint(1, 50),
            'block_shuffle_seed': random.randint(1, 1000000),
            'color_transform': [random.random() for _ in range(3)],
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    def _split_into_blocks(self, image_array: np.ndarray) -> list:
        """Split image into blocks for block-level manipulation"""
        blocks = []
        height, width = image_array.shape[:2]
        
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                block = image_array[i:i+self.block_size, j:j+self.block_size]
                blocks.append((block, (i, j)))
        return blocks

    def _reassemble_blocks(self, blocks: list, shape: Tuple) -> np.ndarray:
        """Reassemble blocks back into a complete image"""
        height, width = shape[:2]
        result = np.zeros(shape, dtype=np.uint8)
        
        for block, (i, j) in blocks:
            h, w = block.shape[:2]
            result[i:i+h, j:j+w] = block
        return result

    def _transform_pixels(self, image_array: np.ndarray, key: dict, encrypt: bool = True) -> np.ndarray:
        """Apply pixel-level transformations"""
        shift = key['pixel_shift'] if encrypt else -key['pixel_shift']
        color_transform = key['color_transform']
        
        # Apply pixel shift
        result = (image_array.astype(np.int16) + shift) % 256
        
        # Apply color transformation
        if len(image_array.shape) == 3:  # Color image
            for i in range(3):  # For each color channel
                if encrypt:
                    result[:,:,i] = (result[:,:,i] * (1 + color_transform[i])) % 256
                else:
                    result[:,:,i] = (result[:,:,i] / (1 + color_transform[i])) % 256
                    
        return result.astype(np.uint8)

    def _shuffle_blocks(self, blocks: list, key: dict, encrypt: bool = True) -> list:
        """Shuffle or unshuffle blocks based on the key"""
        random.seed(key['block_shuffle_seed'])
        indices = list(range(len(blocks)))
        
        if encrypt:
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
        else:
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
            # Create reverse mapping for decryption
            shuffled_indices = [shuffled_indices.index(i) for i in indices]
            
        return [blocks[i] for i in shuffled_indices]

    def process_image(self, input_path: str, encrypt: bool = True, key: dict = None) -> Tuple[np.ndarray, dict]:
        """Main method to process (encrypt/decrypt) an image"""
        # Load and convert image
        image = Image.open(input_path)
        image_array = np.array(image)
        
        # Generate or use provided key
        if encrypt and key is None:
            key = self.generate_key()
        elif not encrypt and key is None:
            raise ValueError("Decryption key is required!")
            
        # Split image into blocks
        blocks = self._split_into_blocks(image_array)
        
        # Apply transformations
        if encrypt:
            # Shuffle blocks
            blocks = self._shuffle_blocks(blocks, key, encrypt=True)
            # Transform pixels
            processed_array = self._transform_pixels(image_array, key, encrypt=True)
        else:
            # Reverse transformations for decryption
            processed_array = self._transform_pixels(image_array, key, encrypt=False)
            blocks = self._shuffle_blocks(blocks, key, encrypt=False)
            
        # Reassemble image
        result = self._reassemble_blocks(blocks, image_array.shape)
        
        return result, key

def save_key(key: dict, filename: str):
    """Save encryption key to a file"""
    with open(filename, 'w') as f:
        for k, v in key.items():
            f.write(f"{k}:{v}\n")

def load_key(filename: str) -> dict:
    """Load encryption key from a file"""
    key = {}
    with open(filename, 'r') as f:
        for line in f:
            k, v = line.strip().split(':')
            if k in ['pixel_shift', 'block_shuffle_seed']:
                key[k] = int(v)
            elif k == 'color_transform':
                key[k] = [float(x) for x in v.strip('[]').split(',')]
            else:
                key[k] = v
    return key

def main():
    """Main execution function with user interface"""
    encryptor = PixelEncryption()
    
    print("=== Advanced Image Encryption Tool ===")
    while True:
        choice = input("\n1. Encrypt Image\n2. Decrypt Image\n3. Exit\nChoice: ")
        
        if choice == '3':
            break
            
        input_path = input("Enter image path: ").strip('"')
        
        if choice == '1':
            # Encrypt
            result, key = encryptor.process_image(input_path, encrypt=True)
            output_path = f"encrypted_{os.path.basename(input_path)}"
            key_path = f"key_{key['timestamp']}.txt"
            
            # Save encrypted image and key
            Image.fromarray(result).save(output_path)
            save_key(key, key_path)
            
            print(f"\nEncryption successful!")
            print(f"Encrypted image saved as: {output_path}")
            print(f"Encryption key saved as: {key_path}")
            print("\nKEEP YOUR KEY FILE SAFE! You'll need it for decryption.")
            
        elif choice == '2':
            # Decrypt
            key_path = input("Enter key file path: ").strip('"')
            key = load_key(key_path)
            
            result, _ = encryptor.process_image(input_path, encrypt=False, key=key)
            output_path = f"decrypted_{os.path.basename(input_path)}"
            
            # Save decrypted image
            Image.fromarray(result).save(output_path)
            print(f"\nDecryption successful!")
            print(f"Decrypted image saved as: {output_path}")

if __name__ == "__main__":
    main()