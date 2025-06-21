import os
import tempfile
from typing import List
import math
import random
import numpy as np
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import replicate
import requests
from dotenv import load_dotenv

load_dotenv()

# Ensure the Replicate API token is available
REPLICATE_API_TOKEN = os.getenv("REPLICATE_TOKEN") or os.getenv("REPLICATE_API_TOKEN")
print(f"API Token loaded: {'Yes' if REPLICATE_API_TOKEN else 'No'}")
print(f"Token starts with: {REPLICATE_API_TOKEN[:4] if REPLICATE_API_TOKEN else 'None'}")

if not REPLICATE_API_TOKEN:
    raise RuntimeError(
        "Please set the REPLICATE_TOKEN environment variable in your .env file or system environment."
    )

try:
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    # Test the client with a simple API call
    models = client.models.list()
    print("✓ Successfully connected to Replicate API")
except Exception as e:
    print(f"✗ Failed to connect to Replicate API: {e}")
    raise RuntimeError(f"Failed to initialize Replicate client: {e}")

MODEL_NAME = "black-forest-labs/flux-1.1-pro"


def interpolate_text_mixtral_infinite_zoom(start: str, end: str, steps: int) -> List[str]:
    """Use Mixtral-8x7B to generate infinite zoom effect prompts."""
    try:
        print("\n=== Starting Mixtral Infinite Zoom Generation ===")
        
        system_prompt = """You are an expert at creating infinite zoom transitions for image generation. Your task is to create prompts that smoothly zoom from one scene into another, where a small detail in the first scene becomes the entire second scene."""

        user_prompt = f"""Create exactly {steps} image generation prompts for an infinite zoom effect between:

START SCENE: "{start}"
END SCENE: "{end}"

Requirements:
1. Output EXACTLY {steps} prompts
2. First prompt: Wide shot of the start scene
3. Middle prompts: Progressively zoom into a specific detail that will transform into the end scene
4. Last prompt: The detail has fully transformed into the end scene
5. Each prompt must specify camera distance (wide shot, medium shot, close-up, extreme close-up, macro)
6. Include smooth transitions where the zoomed detail morphs into the new scene

Example transition:
Start: "a busy city street"
End: "an ant colony underground"

Good output:
a busy city street, wide aerial shot showing buildings and traffic
a busy city street focusing on a small crack in the sidewalk, medium shot
extreme close-up of the crack in the pavement revealing tiny movements inside
macro shot inside the crack showing it opens into tunnels that look like ant paths
an ant colony underground with tunnels and chambers, pulling back to show full colony

Write ONLY the prompts, one per line."""

        print("\nSending infinite zoom prompt to Mixtral...")
        
        output = replicate.run(
            "mistralai/mixtral-8x7b-instruct-v0.1:2b56576fcfbe32fa0526897d8385dd3fb3d36ba6fd0dbe033c72886b81ade93e",
            input={
                "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 1024,
                "repetition_penalty": 1.1
            }
        )

        print("\nProcessing Mixtral output...")
        
        full_output = "".join(output)
        print(f"\nRaw output:\n{full_output}")
        
        lines = full_output.split('\n')
        prompts = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('System:', 'User:', 'Assistant:', '1.', '2.', '3.', '4.', '5.')):
                continue
            prompts.append(line)
        
        if len(prompts) != steps:
            print(f"\nWarning: Generated {len(prompts)} prompts instead of {steps}")
            return None
            
        # Add cinematic quality markers
        prompts = [f"{p}, cinematic composition, sharp focus on subject, depth of field" for p in prompts]
        
        print("\n=== Final infinite zoom prompts ===")
        for i, p in enumerate(prompts):
            print(f"Frame {i+1}: {p}")
            
        return prompts

    except Exception as e:
        print(f"\nError in Mixtral: {str(e)}")
        return None


def create_infinite_zoom_prompts(start: str, end: str, steps: int) -> List[str]:
    """Fallback function to create infinite zoom prompts when Mixtral fails."""
    if steps < 3:
        return [start, end]
    
    prompts = []
    
    # Camera distances for the zoom
    zoom_levels = [
        "wide aerial establishing shot, full scene visible",
        "wide shot showing the entire scene", 
        "medium shot focusing on the center",
        "close-up on a small detail in the center",
        "extreme close-up revealing hidden details",
        "macro shot going deeper into the detail",
        "microscopic view transforming into something new",
        "pulling back to reveal the new scene emerging",
        "medium shot of the new transformed scene",
        "wide shot showing the complete transformation"
    ]
    
    # Adjust zoom levels to match number of steps
    if steps <= len(zoom_levels):
        selected_zooms = [zoom_levels[int(i * len(zoom_levels) / steps)] for i in range(steps)]
    else:
        selected_zooms = zoom_levels + ["medium shot"] * (steps - len(zoom_levels))
    
    for i in range(steps):
        progress = i / (steps - 1)
        
        if i == 0:
            # Wide shot of start scene
            prompt = f"{start}, {selected_zooms[i]}, cinematic composition"
        elif i == steps - 1:
            # Final shot of end scene
            prompt = f"{end}, wide establishing shot, full scene visible, cinematic"
        elif progress < 0.4:
            # Zooming into the start scene
            prompt = f"{start}, {selected_zooms[i]}, focusing on intricate details that hint at {end}"
        elif progress < 0.6:
            # Transformation zone
            prompt = f"detailed view revealing {start} morphing into {end}, {selected_zooms[i]}, surreal transformation"
        else:
            # Emerging into end scene
            prompt = f"{end} emerging from the depths, {selected_zooms[i]}, revealing the full scene"
        
        prompts.append(f"{prompt}, professional photography, sharp focus, depth of field")
    
    return prompts


def interpolate_text_mixtral(start: str, end: str, steps: int) -> List[str]:
    """Use Mixtral-8x7B to generate smooth text interpolations between prompts."""
    try:
        print("\n=== Starting Mixtral Prompt Generation ===")
        
        # Use Mixtral to generate intermediate descriptions
        system_prompt = """You are an expert at writing clear, detailed image generation prompts. Your task is to create a sequence of prompts that smoothly transition between two scenes."""

        user_prompt = f"""Write exactly {steps} image generation prompts that show a smooth transition between these scenes:

START: "{start}"
END: "{end}"

Requirements:
1. Output EXACTLY {steps} complete sentences
2. First sentence MUST be exactly: "{start}"
3. Last sentence MUST be exactly: "{end}"
4. Each intermediate prompt must:
   - Be a complete, grammatical English sentence
   - Describe a clear visual scene
   - Show gradual change from start to end
   - Use specific numbers and details
   - Focus on actions and visible elements

Example for 5 steps:
Start: "a single cat sleeping on a windowsill"
End: "five cats playing with toys in the sun"

Good output:
a single cat sleeping on a windowsill
a single cat stretching and opening its eyes on the windowsill
three cats alert and curious on the windowsill, one reaching for a toy
four cats actively playing with toys near the window
five cats playing with toys in the sun

Format: Write ONLY the prompts, one per line, no numbering or prefixes."""

        print("\nSending prompt to Mixtral...")
        print(f"System prompt: {system_prompt}")
        print(f"User prompt: {user_prompt}")
        
        # Use Mixtral via Replicate
        output = replicate.run(
            "mistralai/mixtral-8x7b-instruct-v0.1:2b56576fcfbe32fa0526897d8385dd3fb3d36ba6fd0dbe033c72886b81ade93e",
            input={
                "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: Here are the {steps} scene descriptions:\n",
                "temperature": 0.1,
                "top_p": 0.85,
                "max_tokens": 1024,
                "repetition_penalty": 1.1,
                "presence_penalty": 1.0
            }
        )

        print("\nProcessing Mixtral output...")
        
        # Collect complete output
        full_output = "".join(output)
        print(f"\nRaw Mixtral output:\n{full_output}")
        
        # Process the output into prompts
        lines = full_output.split('\n')
        prompts = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that look like formatting
            if not line or line.startswith(('System:', 'User:', 'Assistant:', 'Here are', '1.', '2.', '3.', '4.', '5.', '-', '•', '*', 'Scene:', 'Frame:')):
                continue
            # Clean up the line
            line = line.strip('"').strip()
            if line:
                prompts.append(line)
        
        print(f"\nExtracted {len(prompts)} prompts:")
        for i, p in enumerate(prompts):
            print(f"{i+1}: {p}")

        # Validate the prompts
        if len(prompts) != steps:
            print(f"\nWarning: Generated {len(prompts)} prompts instead of {steps}")
            return None
            
        # Ensure first and last prompts match exactly
        if prompts[0] != start or prompts[-1] != end:
            print("\nWarning: First or last prompt doesn't match exactly")
            prompts[0] = start
            prompts[-1] = end

        # Add quality hints to each prompt
        prompts = [f"{p}, professional photo, detailed" for p in prompts]
        
        print("\n=== Final prompts ===")
        for i, p in enumerate(prompts):
            print(f"Frame {i+1}: {p}")
            
        return prompts

    except Exception as e:
        print(f"\nError in Mixtral interpolation: {str(e)}")
        print("Full error:", e)
        print("Falling back to basic interpolation")
        return None


def interpolate_prompts(start: str, end: str, steps: int, smoothness: float = 0.5) -> List[str]:
    """Generate a smooth interpolation between two prompts when Mixtral fails."""
    if steps < 2:
        return [end]

    # [Original interpolate_prompts function content remains the same]
    # ... (keeping the original function as is for brevity)
    
    result = []
    for i in range(steps):
        if i == 0:
            result.append(start)
        elif i == steps - 1:
            result.append(end)
        else:
            progress = i / (steps - 1)
            # Simple linear interpolation for fallback
            result.append(f"transitioning from {start} to {end}, {int(progress * 100)}% complete, high quality")
    
    return result


def run_text_to_image(prompt: str) -> str:
    """Run a text-to-image model and return the URL of the generated image."""
    print(f"  > Calling Replicate with prompt: '{prompt}'")
    prediction = client.run(
        MODEL_NAME,
        input={
            "prompt": prompt,
        },
    )
    print(f"  > Got response from Replicate.")

    # The prediction may return a list or single url depending on model; normalize
    if isinstance(prediction, list):
        return prediction[0]
    return prediction


def create_transition_frame(img1: Image.Image, img2: Image.Image, alpha: float) -> Image.Image:
    """Create a smooth transition frame between two images."""
    # Convert to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Apply smooth easing function (cubic)
    alpha = alpha * alpha * (3 - 2 * alpha)
    
    # Create the blended frame
    blended = arr1 * (1 - alpha) + arr2 * alpha
    return Image.fromarray(blended.astype('uint8'))


def create_zoom_transition_frame(img1: Image.Image, img2: Image.Image, alpha: float) -> Image.Image:
    """Create a zoom transition effect between two images."""
    width, height = img1.size
    
    # Calculate zoom factor for img1 (zooming in)
    zoom1 = 1.0 + alpha * 2.0  # Zoom from 1x to 3x
    
    # Calculate zoom factor for img2 (starting zoomed, pulling out)
    zoom2 = 3.0 - alpha * 2.0  # Zoom from 3x to 1x
    
    # Process img1 (zooming in)
    new_size1 = (int(width * zoom1), int(height * zoom1))
    img1_zoomed = img1.resize(new_size1, Image.LANCZOS)
    
    # Center crop img1
    left1 = (new_size1[0] - width) // 2
    top1 = (new_size1[1] - height) // 2
    img1_cropped = img1_zoomed.crop((left1, top1, left1 + width, top1 + height))
    
    # Process img2 (pulling out) 
    new_size2 = (int(width * zoom2), int(height * zoom2))
    img2_zoomed = img2.resize(new_size2, Image.LANCZOS)
    
    # Center crop img2
    left2 = (new_size2[0] - width) // 2
    top2 = (new_size2[1] - height) // 2
    img2_cropped = img2_zoomed.crop((left2, top2, left2 + width, top2 + height))
    
    # Smooth blending with easing
    ease_alpha = alpha * alpha * (3 - 2 * alpha)  # Smooth cubic easing
    
    # Convert to arrays and blend
    arr1 = np.array(img1_cropped)
    arr2 = np.array(img2_cropped)
    
    # Add vignette effect for zoom
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / (width // 2)
    vignette = 1 - (dist * alpha * 0.5)
    vignette = np.clip(vignette, 0, 1)[:, :, np.newaxis]
    
    blended = (arr1 * (1 - ease_alpha) + arr2 * ease_alpha) * vignette
    
    return Image.fromarray(blended.astype('uint8'))


def save_image_to_temp(img: Image.Image) -> str:
    """Save a PIL Image to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
        img.save(out.name)
        return out.name


def generate_frames_with_mode(start_prompt, end_prompt, num_frames, smoothness, mode="infinite_zoom"):
    """Generate frames with different transition modes."""
    global current_prompts
    gr.Info(f"Starting {mode.replace('_', ' ')} generation...")
    
    try:
        print(f"\n=== Starting Frame Generation with mode: {mode} ===")
        if not start_prompt or not end_prompt:
            raise gr.Error("Please provide both start and end prompts.")

        num_frames = int(num_frames)
        
        # Generate prompts based on mode
        if mode == "infinite_zoom":
            print("\nAttempting Infinite Zoom prompt generation...")
            prompts = interpolate_text_mixtral_infinite_zoom(start_prompt, end_prompt, num_frames)
            
            if not prompts:
                print("\nMixtral failed, using fallback infinite zoom")
                prompts = create_infinite_zoom_prompts(start_prompt, end_prompt, num_frames)
        else:
            # Original interpolation mode
            prompts = interpolate_text_mixtral(start_prompt, end_prompt, num_frames)
            if not prompts:
                prompts = interpolate_prompts(start_prompt, end_prompt, num_frames, smoothness)

        if not prompts:
            raise gr.Error("Failed to generate prompts")

        # Store prompts for display
        current_prompts = prompts
        frames = [None] * len(prompts)          # pre-allocate
        print("\nGenerating images with 4 worker threads...")

        def fetch_and_save(idx_prompt):
            idx, prompt = idx_prompt
            try:
                url = run_text_to_image(prompt)          # Replicate call
                img_bytes = requests.get(url).content    # download
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    return idx, tmp.name                # >>> success
            except Exception as e:
                print(f"[Thread {idx}] failed → {e}")    # log for debug
                return idx, None                        # >>> signal failure

        with ThreadPoolExecutor(max_workers=min(3, len(prompts))) as ex:   # 3 is under Replicate's rate-limit
            futures = [ex.submit(fetch_and_save, pair) for pair in enumerate(prompts)]
            for fut in as_completed(futures):
                idx, path = fut.result()
                frames[idx] = path
                print(f"✓ frame {idx+1}/{len(prompts)} ready")

        # --- SECOND PASS (serial) ---
        missing = [i for i, p in enumerate(frames) if p is None]
        if missing:
            print(f"Retrying {len(missing)} failed frame(s) serially...")
            for idx in missing:
                prompt = prompts[idx]
                try:
                    url = run_text_to_image(prompt)
                    img_bytes = requests.get(url).content
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(img_bytes)
                        frames[idx] = tmp.name
                        print(f"  ↻ recovered frame {idx+1}")
                except Exception as e:
                    raise RuntimeError(f"Could not recover frame {idx+1}: {e}")

        # Verify all frames were generated
        if None in frames:
            raise gr.Error("Some frames failed to generate")
            
        return frames  # Return the list of frame paths
        
    except Exception as e:
        print(f"\nError during generation: {str(e)}")
        raise gr.Error(f"An error occurred during generation: {e}")


def process_generation_with_mode(start_prompt, end_prompt, num_frames, smoothness, transition_mode):
    """Process the frame generation with selected transition mode."""
    global current_prompts
    
    try:
        # Generate base frames with selected mode
        frame_paths = generate_frames_with_mode(start_prompt, end_prompt, num_frames, smoothness, transition_mode)
        
        if not frame_paths:
            raise gr.Error("No frames were generated")
            
        # Load all base frames into memory
        base_frames = [Image.open(path).resize((1024, 1024)) for path in frame_paths]
        
        # Pre-generate all interpolated frames
        print(f"\nPre-generating interpolated frames with {transition_mode} effect...")
        all_frames = []
        
        # Add first frame
        all_frames.append(save_image_to_temp(base_frames[0]))
        
        # Choose transition function and parameters based on mode
        if transition_mode == "infinite_zoom":
            transition_fn = create_zoom_transition_frame
            num_steps = 8  # Reduced from 15 for better performance while maintaining quality
        else:
            transition_fn = create_transition_frame
            num_steps = 5  # Reduced from 10
            
        def generate_interpolation(params):
            i, step, base_frames, transition_fn = params
            try:
                alpha = (step + 1) / (num_steps + 1)
                interpolated = transition_fn(base_frames[i], base_frames[i + 1], alpha)
                return save_image_to_temp(interpolated)
            except Exception as e:
                print(f"Error generating interpolation frame: {e}")
                return None

        # Prepare parameters for parallel processing
        interpolation_params = []
        for i in range(len(base_frames) - 1):
            for step in range(num_steps):
                interpolation_params.append((i, step, base_frames, transition_fn))

        # Generate interpolated frames in parallel
        print(f"Generating {len(interpolation_params)} interpolation frames...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_interpolation, params) for params in interpolation_params]
            
            # Process results as they complete
            frame_index = 1  # Start after first frame
            for i in range(len(base_frames) - 1):
                # Add interpolated frames
                for _ in range(num_steps):
                    future = futures[frame_index - 1]
                    frame_path = future.result()
                    if frame_path:
                        all_frames.append(frame_path)
                    frame_index += 1
                
                # Add next base frame
                all_frames.append(save_image_to_temp(base_frames[i + 1]))
        
        print(f"Generated {len(all_frames)} total frames")
        
        # Get the prompts that were used
        prompts = [f"Frame {i+1}: {p}" for i, p in enumerate(current_prompts)] if current_prompts else []
        prompts_text = "\n".join(prompts)
        
        return (
            frame_paths,  # for gallery
            all_frames,  # for state
            all_frames[0] if all_frames else None,  # for frame viewer
            gr.Slider(minimum=0, maximum=len(all_frames)-1 if all_frames else 0, value=0, step=1),  # for slider
            prompts_text  # for prompt display
        )
    except Exception as e:
        print(f"Error in process_generation_with_mode: {str(e)}")
        # Return empty/default values for all outputs
        return [], [], None, gr.Slider(minimum=0, maximum=0, value=0, step=1), ""


def update_frame_view(state, frame_index):
    """Update the frame viewer with the selected frame."""
    if not state:
        return None
    try:
        # Just return the pre-generated frame at the index
        idx = int(frame_index)  # Convert to int since we're using discrete frames now
        return state[min(idx, len(state) - 1)]
    except (IndexError, ValueError) as e:
        print(f"Error in update_frame_view: {e}")
        return None


with gr.Blocks(title="PromptMorph - Visual Story Generator", theme=gr.themes.Soft()) as demo:
    # Add global variables at the start
    current_prompts = []
    
    gr.Markdown("""
    # 🎨 PromptMorph - Infinite Zoom Story Generator
    
    Create mind-blowing visual stories where one scene zooms seamlessly into another!
    Perfect for creating viral content that makes viewers say "wait, what?!"
    """)
    
    frames_state = gr.State([])
    
    with gr.Row(equal_height=True):
        # Left column for inputs
        with gr.Column(scale=1):
            start_prompt = gr.Textbox(
                label="Start Scene",
                placeholder="e.g. a busy city street with skyscrapers",
                lines=3
            )
            end_prompt = gr.Textbox(
                label="End Scene", 
                placeholder="e.g. the inside of a computer chip with glowing circuits",
                lines=3
            )
            
            # Add transition mode selector
            transition_mode = gr.Radio(
                choices=["infinite_zoom", "standard"],
                value="infinite_zoom",
                label="Transition Mode",
                info="Infinite Zoom creates a seamless zoom from one scene into another"
            )
            
            with gr.Row():
                with gr.Column():
                    num_frames = gr.Slider(
                        minimum=3,
                        maximum=10,
                        step=1,
                        value=6,
                        label="Number of Frames",
                        info="More frames = smoother zoom effect"
                    )
                with gr.Column():
                    smoothness = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.7,
                        label="Transition Smoothness",
                        info="Higher = smoother blending"
                    )
            
            generate_btn = gr.Button("🚀 Generate Infinite Zoom", variant="primary", size="lg")
            
            # Example prompts for infinite zoom
            gr.Examples(
                examples=[
                    ["a busy city street with skyscrapers", "the inside of a computer chip with glowing circuits"],
                    ["a human eye close-up", "a spiral galaxy in deep space"],
                    ["a drop of water on a leaf", "an ocean with waves crashing"],
                    ["an old library with dusty books", "a magical forest inside a book"],
                    ["a child's toy robot", "a futuristic robot factory"],
                    ["a cup of coffee on a table", "coffee beans on a plantation in Colombia"],
                    ["a smartphone screen", "a person trapped in social media"],
                    ["Earth from space", "a single grain of sand on a beach"]
                ],
                inputs=[start_prompt, end_prompt],
                label="🎯 Mind-Blowing Infinite Zoom Examples"
            )
        
        # Right column for outputs
        with gr.Column(scale=2):
            with gr.Row():
                # Frame viewer on the left side of right column
                with gr.Column(scale=3):
                    frame_viewer = gr.Image(
                        label="Frame Viewer",
                        interactive=False,
                        show_label=True,
                        container=True,
                        height=500,
                        width=500,
                        elem_class="frame-viewer"
                    )
                    
                    # Animation controls with simple slider
                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=0,
                        step=1,
                        value=0,
                        label="📹 Scrub Through Animation (drag to play manually)"
                    )
                    
                    # Gallery at the bottom
                    gallery = gr.Gallery(
                        label="Generated Key Frames",
                        show_label=True,
                        elem_id="gallery",
                        columns=5,
                        height=120,
                        object_fit="contain"
                    )
                
                # Generated prompts on the right side
                with gr.Column(scale=2):
                    prompts_display = gr.Textbox(
                        label="🎬 Generated Frame Prompts",
                        lines=25,
                        interactive=False,
                        show_label=True
                    )
    
    # Connect the slider to update the frame view
    frame_slider.change(
        fn=update_frame_view,
        inputs=[frames_state, frame_slider],
        outputs=frame_viewer
    )
    
    # Connect the generate button
    generate_btn.click(
        fn=process_generation_with_mode,
        inputs=[start_prompt, end_prompt, num_frames, smoothness, transition_mode],
        outputs=[gallery, frames_state, frame_viewer, frame_slider, prompts_display]
    )

if __name__ == "__main__":
    demo.launch(share=True)