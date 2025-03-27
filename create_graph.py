from MIND import compute_MIND
import os

surf_dir = r'\\wsl$\Ubuntu-22.04\home\efekan\adni'
features = ['CT', 'Vol', 'SA', 'MC', 'SD']
parcellation = 'aparc'

# Get all subject directories, excluding 'fsaverage' and any shell scripts
subject_ptids = os.listdir(surf_dir)
subject_ptids = [i for i in subject_ptids if (i != 'fsaverage' and '.sh' not in i)]

for id in subject_ptids:
    subject_dir = os.path.join(surf_dir, id)
    surf_files_dir = os.path.join(subject_dir, 'surf')
    
    # Skip if surf_files_dir doesn't exist or is empty
    if not os.path.isdir(surf_files_dir) or not os.listdir(surf_files_dir):
        continue

    print(subject_dir, id)

    try:
        mind_graph = compute_MIND(subject_dir, features, parcellation)
        output_path = f'mind_adni1_bl_tiny/graphs/{id}.csv'
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        mind_graph.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error processing {id}: {e}")

print("Done!")