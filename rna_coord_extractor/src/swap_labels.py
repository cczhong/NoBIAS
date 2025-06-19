#!/usr/bin/env python
# coding: utf-8

# In[7]:


pairs = ['AA', 'AG', 'AC', 'AU', 'GA', 'GG', 'GC', 'GU', 'CA', 'CG', 'CC', 'CU', 'UA', 'UG', 'UC', 'UU']
counterpart_pairs = {'CA': 'AC', 'GA': 'AG', 'GC': 'CG', 'UA': 'AU', 'UC': 'CU', 'UG': 'GU'}
target_classes = ['cWW', 'tWW', 'cWH', 'cHW', 'tWH', 'tHW', 'cWS', 'cSW', 'tWS', 'tSW', 'cHH', 'tHH', 'cHS', 'cSH', 'tHS', 'tSH', 'cSS', 'tSS','Unknown','Neg']

def swap_class(bp, target_class):
    bp = bp.replace('T', 'U')
    #swapped_bp = counterpart_pairs[bp]
    # Check if the base pair has a counterpart
    if bp in counterpart_pairs:
        swapped_bp = counterpart_pairs[bp]
        # Exclude 'Neg' and 'Unknown' classes from swapping
        
            # Swap the base pair
            
        #swapped_bp = counterpart_pairs[bp]
        #if target_class not in ['Neg', 'Unknown']:    
            # Construct the swapped target class
        if len(target_class) >= 3:
            swapped_target_class = target_class[0] + target_class[2] + target_class[1]
        else:
            swapped_target_class = 'Unknown'

        if target_class in ['Neg', 'Unknown']:
            return swapped_bp, target_class
        else:
            return swapped_bp, swapped_target_class
     
    # If no counterpart or class is 'Neg' or 'Unknown', return the original pair and class
    return bp, target_class

"""
# Example usage:
original_bp = 'TA'
original_class = 'Neg'

swapped_bp, swapped_class = swap_class(original_bp, original_class)
print(f"Original: {original_bp}, {original_class}")
print(f"Swapped: {swapped_bp}, {swapped_class}")
"""

