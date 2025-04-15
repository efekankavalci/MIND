from MIND import compute_MIND
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# surf_dir = r'C:\Users\efeka\Desktop\TubitakData\2025.01.03_mri_patients_features\tubitak_fs'
surf_dir = r'\\wsl.localhost\Ubuntu-22.04\home\efekan\adni1_bl_surfaces\ADNI'
features = ['CT', 'Vol', 'SA', 'MC', 'SD']
parcellation = 'aparc'

def process_subject(id):
    subject_dir = os.path.join(surf_dir, id)
    surf_files_dir = os.path.join(subject_dir, 'surf')
    output_path = f'mind_adni1_bl/MIND_filtered_vertices/{id}.csv'

    if not os.path.isdir(surf_files_dir) or not os.listdir(surf_files_dir):
        return f"Skipping {id} (no surf files)."

    if os.path.exists(output_path):
        return f"Skipping {id} (already processed)."

    try:
        mind_graph = compute_MIND(subject_dir, features, parcellation, filter_vertices=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mind_graph.to_csv(output_path, index=False)
        return f"Finished {id}"
    except Exception as e:
        return f"Error processing {id}: {e}"

if __name__ == "__main__":
    subject_ptids = os.listdir(surf_dir)
    subject_ptids = [i for i in subject_ptids if (i != 'fsaverage' and '.sh' not in i)]

    with ProcessPoolExecutor(max_workers=os.cpu_count()-2) as executor:
        futures = [executor.submit(process_subject, ptid) for ptid in subject_ptids]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Subjects"):
            print(future.result())

    print("Done!")


 

# from MIND import compute_MIND
# import os

# # surf_dir = r'\\wsl$\Ubuntu-22.04\home\efekan\adni'
# surf_dir = r'C:\Users\efeka\Desktop\TubitakData\2025.01.03_mri_patients_features\tubitak_fs'
# features = ['CT', 'Vol', 'SA', 'MC', 'SD']
# parcellation = 'aparc'

# # Get all subject directories, excluding 'fsaverage' and any shell scripts
# subject_ptids = os.listdir(surf_dir)
# subject_ptids = [i for i in subject_ptids if (i != 'fsaverage' and '.sh' not in i)]

# for id in subject_ptids:
#     print(f"Processing {id}...")
#     subject_dir = os.path.join(surf_dir, id)
#     surf_files_dir = os.path.join(subject_dir, 'surf')
#     output_path = f'tubitak_graphs/{id}.csv'
    
#     # Skip if surf_files_dir doesn't exist or is empty
#     if not os.path.isdir(surf_files_dir) or not os.listdir(surf_files_dir):
#         continue

#     # Skip if output CSV already exists
#     if os.path.exists(output_path):
#         print(f"Skipping {id} (already processed).")
#         continue

#     print(subject_dir, id)

#     try:
#         mind_graph = compute_MIND(subject_dir, features, parcellation)

#         # Make sure output directory exists
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # mind_graph.to_csv(output_path, index=False)
#     except Exception as e:
#         print(f"Error processing {id}: {e}")

# print("Done!")
