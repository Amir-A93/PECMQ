""" ldp_simulator.py
* Input: JSON file of CPR sets [(challenge, feature, response)]
* Output: JSON file of CPR sets with randomized responses [(challenge, feature, randomized_response)]
* Usage: Takes in a JSON file, modifies the response element of each CPR based on LDP-guaranteed 
        randomized response bit flipping while preserving the total number of 1's and 0's
        in one CRP set.
"""

import json
import random
import argparse
from typing import List, Dict

parser = argparse.ArgumentParser(description="LDP Simulator")

# Add arguments for FLIP_PROBABILITY, PRINT_OUT, and LIMIT
parser.add_argument("--flip_probability", type=float, default=0.5, help="Probability of flipping each bit (0.0 to 1.0)")
parser.add_argument("--print_out", type=bool, default=False, help="Whether to print output for verification")
parser.add_argument("--cpack_size", type=int, default=20, help="The size of each Cpack of the challenge")

parser.add_argument("--input_json_file", type=str, default="CRP_set_size_20000_ch_size_64_xor_size_2.json", help="Input JSON file name")
parser.add_argument("--segmented_input_json_file", type=str, default="Segmented_CRP_set_size_20000_ch_size_64_xor_size_2.json", help="Input JSON file name")
parser.add_argument("--output_json_file", type=str, default="LDP_Randomized_CRP_set_size_20000_ch_size_64_xor_size_2.json", help="Output JSON file name")


args = parser.parse_args()

############# Bit Flipper #################
def flip_and_preserve_count(responses: List[float], flip_probability: float = 0.5) -> List[float]:
    """Randomly flips bits while preserving the count of 0's and 1's in the responses."""
    
    num_zeros = responses.count(0.0)
    num_ones = responses.count(1.0)
    
    available_flips = []
    
    for response in responses:
        if random.random() < flip_probability:
            available_flips.append(1.0 if response == 0.0 else 0.0)
        else:
            available_flips.append(response)
    
    current_zeros = available_flips.count(0.0)
    current_ones = available_flips.count(1.0)
    
    while current_zeros > num_zeros:
        index_to_flip = random.choice([i for i, v in enumerate(available_flips) if v == 0.0])
        available_flips[index_to_flip] = 1.0
        current_zeros -= 1
        current_ones += 1
    
    while current_ones > num_ones:
        index_to_flip = random.choice([i for i, v in enumerate(available_flips) if v == 1.0])
        available_flips[index_to_flip] = 0.0
        current_ones -= 1
        current_zeros += 1
    
    return available_flips

############# JSON file processor #################

def process_json(input_file: str, segmented_file: str, output_file: str, cpack_size = 7, flip_probability: float = 0.5) -> None:
    """Processes the JSON file, flips response bits in each group while preserving counts, and writes the output to a new file."""
    segmented_input_file = group_cpack(input_file,segmented_file, cpack_size)
    
    with open(segmented_input_file, 'r') as infile:
        data = json.load(infile)

    for group_key, group_data in data.items():
        if isinstance(group_data, dict):
            responses = []
            for subgroup_key, subgroup_data in group_data.items():
                # print(f"Subgroup key: {subgroup_key}, Type: {type(subgroup_data)}")  
                # print(f"Subgroup data: {subgroup_data}")  
                
                if isinstance(subgroup_data, dict):  
                    if 'response' in subgroup_data: 
                        responses.append(subgroup_data['response']) 
                    else:
                        print("Something went wrong////////////////////////////////")
                        exit()
            
            # Flip the responses within this group while preserving counts
            flipped_responses = flip_and_preserve_count(responses, flip_probability)
            
            for (subgroup_key, subgroup_data), new_response in zip(group_data.items(), flipped_responses):
                subgroup_data['response'] = new_response  # Update the response value

    # Write the modified data to the output JSON file
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    return segmented_input_file, output_file

def group_cpack(input_file: Dict, segmented_file: str, cpack_size: int) -> str:

    with open(input_file, 'r') as infile:
        data = json.load(infile)

    # Ensure that the number of elements is divisible by cpack_size
    assert len(data) % cpack_size == 0, f"Error: The JSON size {len(data)} is not divisible by cpack_size {cpack_size}."

    grouped_data = {}
    current_group = {}
    group_index = 0

    for index, (key, value) in enumerate(data.items()):
        # Add the current challenge-response pair to the current group
        current_group[key] = value
        
        # If the group size reaches cpack_size, store the group and start a new one
        if (index + 1) % cpack_size == 0:
            grouped_data[str(group_index)] = current_group
            current_group = {}
            group_index += 1

    segmented_input_json_file = segmented_file
    with open(segmented_input_json_file, 'w') as outfile:
        json.dump(grouped_data, outfile, indent=4)

    return segmented_input_json_file

############# Helper functions #################

# Verification helper

def count_response_bits(data: Dict, print_out: bool, file_name: str) -> Dict[str, int]:
    """Counts the total number of 0's and 1's in the response fields of the data."""
    group_counts = {}
    

    if print_out and file_name:
        with open(file_name, 'w') as file:
            file.write("Response is as follows:\n")

    for group_key, group_data in data.items():
        total_zeros = 0
        total_ones = 0

        if isinstance(group_data, dict):  

            for subgroup_key, subgroup_data in group_data.items():

                if isinstance(subgroup_data, dict) and 'response' in subgroup_data:
                    response = subgroup_data['response'] 
                    
                    # Count the number of 0's and 1's
                    if response == 0.0:
                        total_zeros += 1
                    elif response == 1.0:
                        total_ones += 1
                    if print_out and file_name:
                        with open(file_name, 'a+') as file:
                            file.write(f"Group {group_key} - Subgroup {subgroup_key}: {response}\n")
                else:
                    assert(f"Error: No 'response' found in subgroup {subgroup_key} of group {group_key}.")
        group_counts[group_key] = {'zeros': total_zeros, 'ones': total_ones}
    return group_counts



def verify_response_counts(input_file: str, output_file: str, print_out: bool) -> bool:
    """Verifies that the number of 1's and 0's in each group of the 'response' fields of the input and output JSON files are the same."""
    
    # Load input and output JSON files
    with open(input_file, 'r') as infile:
        input_data = json.load(infile)
    
    with open(output_file, 'r') as outfile:
        output_data = json.load(outfile)

    input_group_counts = count_response_bits(input_data, print_out, "original_grouped_CRP_responses.txt")
    output_group_counts = count_response_bits(output_data, print_out, "modified_grouped_CRP_responses.txt")

    different_bits_count = 0
    counts_match = True  # Flag to track whether counts match across groups

    # individual sub group check
    for group_key, group_data in input_data.items():
        input_group_count = input_group_counts[group_key]
        output_group_count = output_group_counts[group_key]
        # print(input_group_count, output_group_count)
        
        if not output_group_count or input_group_count != output_group_count:
            print(f"Mismatch in group {group_key}:")
            print(f"Input counts: {input_group_count}, Output counts: {output_group_count}")
            counts_match = False
        else:
            print(f"Group {group_key} for input and output counts are the same: {input_group_count}, {output_group_count}")
            assert (input_group_count == output_group_count)
        subgroup_different_bits_count = 0
        

        for key, value in group_data.items():
            if isinstance(value, dict) and 'response' in value:
                original_response = value['response']
                modified_response = output_data[group_key][key]['response']

                if original_response != modified_response:
                    different_bits_count += 1
                    subgroup_different_bits_count += 1
        print(f"Group {group_key} has {subgroup_different_bits_count} of different bits after randomization.")

    # total number check: 
    total_input_zeros = sum(group['zeros'] for group in input_group_counts.values())
    total_input_ones = sum(group['ones'] for group in input_group_counts.values())

    total_output_zeros = sum(group['zeros'] for group in output_group_counts.values())
    total_output_ones = sum(group['ones'] for group in output_group_counts.values())

    print(f"\nTotal zeros in input: {total_input_zeros}, Total ones in input: {total_input_ones}")
    print(f"Total zeros in output: {total_output_zeros}, Total ones in output: {total_output_ones}")

    # Check if total zeros and ones match between input and output
    if total_input_zeros != total_output_zeros or total_input_ones != total_output_ones:
        print("Mismatch in total number of zeros or ones between input and output.\n")
        counts_match = False
    else:
        print("Total number of zeros and ones match between input and output.")
    
    print(f"Total number of different bits after randomization: {different_bits_count}")

    return counts_match

# if __name__ == "__main__":

#     input_json_file = args.input_json_file
#     output_json_file = args.output_json_file
    
#     # # Process the JSON file and randomly shuffle the challenge bits while preserving counts
#     segmented_input_json_file, output_json_file = process_json(input_json_file, output_json_file)

#     # # Verify that the number of 1's and 0's are preserved after LDP random flipping
#     if verify_response_counts(segmented_input_json_file, output_json_file, args.print_out):
#         print("The number of 1's and 0's in the responses is preserved.")
#     else:
#         print("Error: Mismatch in number of 0's and 1's detected.")
