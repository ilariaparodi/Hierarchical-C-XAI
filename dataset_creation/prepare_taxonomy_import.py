import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(SCRIPT_DIR, 'annotations.json')
# 1. Load the annotations you uploaded
with open(json_path, 'r') as f:
    original_data = json.load(f)

label_studio_import = []

for item in original_data:
    # 2. Format the URL to work with local file serving (forcing forward slashes)
    safe_path = item['image'].replace('\\', '/')
    local_image_url = f"/data/local-files/?d={safe_path}"
    
    # 3. Build the specific tree pathways for the Taxonomy UI
    # We capitalize the final_class and coarse_concept to match the XML UI precisely
    final_class_ui = item['final_class'].capitalize()
    coarse_concept_ui = item['coarse_concept'].capitalize()
    
    taxonomy_selections = []
    
    # Map every active fine concept to its full directory path
    for fine_concept in item['active_fine_concepts']:
        path = [final_class_ui, coarse_concept_ui, fine_concept]
        taxonomy_selections.append(path)
        
    # 4. Construct the Label Studio task
    ls_task = {
        "data": {
            "image": local_image_url
        },
        "predictions": [
            {
                "model_version": "baseline_class_assumptions",
                "result": [
                    {
                        "from_name": "fine_concepts",
                        "to_name": "image",
                        "type": "taxonomy",
                        "value": {
                            "taxonomy": taxonomy_selections
                        }
                    }
                ]
            }
        ]
    }
    label_studio_import.append(ls_task)

# 5. Save the final payload
with open('label_studio_taxonomy_import.json', 'w') as f:
    json.dump(label_studio_import, f, indent=2)

print(f"Successfully prepared {len(label_studio_import)} tasks for Label Studio.")
print("File saved as: label_studio_taxonomy_import.json")