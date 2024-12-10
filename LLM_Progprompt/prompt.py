Background prompt:
Given a robot on a table with some objects, we provide a code-based prompt describing the object list, primitive actions, and a task. The robot can only operate in discrete primitive action with objects. Write in code the step-by-step planner for completing the task. Use as the minimum number of primitive actions as possible to accomplish the task.

Input Prompt:
Objects = ['blue patch, 'green patch', 'robotarm','cracker box' ] #Detected from VLM
Task = "Put the cracker box onto the blue patch" #Provided by the user
primitive_action_list = [
                            "Locate",
                            "Cut",
                            "Bend",
                            "Reorient", 
                            "Slide",
                            "Push",
                            "Pull",
                            "Drop",
                            "Place",
                            "Pick",
                            "Turn",
                            "Fold",
                            "Unfold",
                            "Rotate"
                            
                            ] #Pre-defined actions

def locate(Objects, target1):
    try:
        index = Objects.index(target1)
        return f"Located {target1} at index {index}"
    except ValueError:
        return f"{target1} not found"

def cut(Objects, target1, target2):
    Objects.cut(target1,target2)
    return f"target2 cutted by target 1"

def bend(Objects, target1):
    new_target1 = Objects.bend(target1)
    if new_target1 != target1:
        return True
    else:
        return False

def reorient(Objects, target1):
    new_target1 = Objects.reorient(target1)
    if new_target1.angle != target1.angle:
        return True
    else:
        return False
def rotate(Objects, target1):
    new_target1 = Objects.reorient(target1)
    if new_target1.angle != target1.angle:
        return True
    else:
        return False

def slide(Objects, target1):
    new_target1 = Objects.slide(target1)
    if new_target1.pose != target1.pose:
        return True
    else:
        return False

def push(Objects, target1):
    new_target1 = Objects.push(target1)
    if new_target1.pose != target1.pose:
        return True
    else:
        return False

def pull(Objects, target1):
    new_target1 = Objects.pull(target1)
    if new_target1.pose != target1.pose:
        return True
    else:
        return False

def drop(Objects, target1):
    new_target1 = Objects.drop(target1)
    if new_target1.pose != target1.pose:
        return True
    else:
        return False

def place(Objects, target1):
    if target1.collision == False:
        return True

def turn(obj_list, index):
    # Assuming turning means reversing the item
    obj_list[index] = obj_list[index][::-1] if isinstance(obj_list[index], str) else obj_list[index]
    return obj_list

def fold(obj_list):
    half = len(obj_list)//2
    return obj_list[:half]

def unfold(obj_list, additional_items):
    return obj_list + additional_items

Output prompt:
#Describle the task
#Write all the output prompt into one code script, with using the minimum number steps.

#Step 1
Step 1 = <Primitive Action><Object> 
Verification Condition 1=" " #Given a sentence description (for evaluating a VLM on the input scene) on how can the user determine if the step has been executed successfully?

#Step 2
Step 2 = <Primitive Action><Object> 
Verification Condition 2=" "