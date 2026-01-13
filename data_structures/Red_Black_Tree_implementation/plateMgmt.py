# -*- coding: utf-8 -*-
import sys
import string
import random
import os
from typing import Type, TypeVar, Iterator



T = TypeVar('T', bound='Node')
# Node creation
class Node():

    def __init__(self: T, key: str) -> None:
        self._key = key
        self.parent = None
        self.left = None
        self.right = None
        self._color = 1 #[0 = black, 1 = red]
        self.value = None
    
    @classmethod
    def null(nodeClass: Type[T]) -> T:
        node = nodeClass(0)
        node._key = None
        node.set_color("black")
        return node



    def set_color(self: T, color: str) -> None:
        allowed_colors = ["red","black"]

        if not color in allowed_colors:
            raise Exception("Unknown color")
        
        if color == "black":
            self._color = 0
        elif color == "red":
            self._color = 1

    
    def set_value(self:T, value:int) -> None:
        self.value=value

    def get_color(self: T) -> str:
        if self._color == 0:
            return "black"
        else:
            return "red"
    
    def get_key(self: T) -> str:
        return self._key

    def is_red(self: T) -> bool:
        if self._color == 1:
            return True
        else:
            return False

    def is_black(self: T) -> bool:
        if self._color == 0:
            return True
        else:
            return False

    def is_null(self: T) -> bool:
        return self._key is None

    def is_parent_null(self: T)-> bool:
        if self.parent is None:
            return True
        else:
            return False



T = TypeVar('T', bound='RedBlackTree')


class RedBlackTree():

    """
    __init__ initializes an empty Red-Black Tree.
    set the root and leaf node to null. Size of RBT set to 0
    """
    def __init__(self: T) -> None:
        self.terminal_null = Node.null()
        self.size = 0
        self.root = self.terminal_null
       


    """
    get_root is retuning the root node of the RBT
    """
    def get_root(self: T) -> Node:
        return self.root
  

    def inorder(self: T) -> list:
        return self._collect_nodes_inorder(self.root)


    def _collect_nodes_inorder(self: T, node: Node) -> list:
       
        output = []
        if not node.is_null():
            left = self._collect_nodes_inorder(node.left)
            right = self._collect_nodes_inorder(node.right)
            output.extend(left)
            output.extend([node])
            output.extend(right)
        return output
    
    """
    Search the tree
    Parameters: Node, Key
    If node is null of key is =node , return node.
    If key < node key  , move to left subtree, otherwise move to rightree
    
    """

    # Search the tree
    def _seach_rbt(self: T, node: Node, key: str) -> Node:
        if node.is_null() or key == node.get_key():
            return node

        if key < node.get_key():
            return self._seach_rbt(node.left, key)
        return self._seach_rbt(node.right, key)


    def _handle_red_parent_del(self: T, sibling_node: Node, fix_node: Node) -> None:
        if sibling_node.is_red():
            sibling_node.set_color("black")
            fix_node.parent.set_color("red")

            if fix_node == fix_node.parent.left:
                self.rotate(fix_node.parent,"left")
                sibling_node = fix_node.parent.right
            else:
                slef.rotate(fix_node.parent,"right")
                sibling_node = fix_node.parent.left
        else:
            return

    def _handle_blac_sibling_del(self: T, sibling_node: Node, fix_node: Node) -> Node:
        if sibling_node.right.is_black():
            sibling_node.left.set_color("black")
            sibling_node.set_color("red")
            self.rotate(sibling_node,"right")
            sibling_node = fix_node.parent.right

        elif sibling_node.left.is_black():
            sibling_node.right.set_color("black")
            sibling_node.set_color("red")
            self.rotate(sibling_node,"left")
            sibling_node = fix_node.parent.left

    """
        Fixes the Red-Black Tree properties after a node deletion. Ensures that the tree
        maintains its balance and color properties.

        Args:x(Node): the node which causes the imbalance 
        Returns:None: The tree is modified .
        Four different scinarios are taken care of 
    """
    # Balancing the tree after deletion
    def delete_fix(self: T, fix_node: Node) -> None:
        while fix_node != self.root and fix_node.is_black():
            if fix_node == fix_node.parent.left:
                sibling_node = fix_node.parent.right
                if sibling_node.is_red():
                   self._handle_red_parent_del(sibling_node,fix_node)

                if sibling_node.left.is_black() and sibling_node.right.is_black():
                    sibling_node.set_color("red")
                    fix_node = fix_node.parent
                else:
                    if sibling_node.right.is_black():
                        self._handle_blac_sibling_del(sibling_node,fix_node)

                    sibling_node.set_color(fix_node.parent.get_color())
                    fix_node.parent.set_color("black")
                    sibling_node.right.set_color("black")
                    self.rotate(fix_node.parent,"left")
                    fix_node = self.root
            else:
                sibling_node = fix_node.parent.left
                if sibling_node.is_red():
                   self._handle_red_parent_del(sibling_node, fix_node)

                if sibling_node.left.is_black() and sibling_node.right.is_black():
                    sibling_node.set_color("red")
                    fix_node = fix_node.parent
                else:
                    if sibling_node.left.is_black():
                         self._handle_blac_sibling_del(sibling_node,fix_node)

                    sibling_node.set_color(fix_node.parent.get_color())
                    fix_node.parent.set_color("black")
                    sibling_node.left.set_color("black")
                    self.rotate(fix_node.parent,"right")
                    fix_node = self.root
        fix_node.set_color("black")

    def _swap_out(self: T, u: Node, v: Node) -> None:
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent


    """
    Deletes a node with the given key from the Red-Black Tree and maintains the tree's balance.
    Arguments: node- the current node in in tree during traversal
    Key: Key of the node to be deleted . It modifies the tree
    """
    
    # Node deletion
    def _find_and_delete(self: T, node: Node, key: str) -> None:
        node_to_del = self.terminal_null
        while not node.is_null():
            if node.get_key() == key:
                node_to_del = node

            if node.get_key() <= key:
                node = node.right
            else:
                node = node.left

        #node was not found
        if node_to_del.is_null():
            return

        temp = node_to_del
        orig_color = temp.get_color()
        if node_to_del.left.is_null():
            # If no left child, just scoot the right subtree up
            cpy_node = node_to_del.right
            self._swap_out(node_to_del, node_to_del.right)
        elif node_to_del.right.is_null():
            # If no right child, just scoot the left subtree up
            cpy_node = node_to_del.left
            self._swap_out(node_to_del, node_to_del.left)
        else:
            temp = self.minimum(node_to_del.right)
            orig_color = temp.get_color()
            cpy_node = temp.right
            if temp.parent == node_to_del:
                cpy_node.parent = temp
            else:
                self._swap_out(temp, temp.right)
                temp.right = node_to_del.right
                temp.right.parent = temp

            self._swap_out(node_to_del, temp)
            temp.left = node_to_del.left
            temp.left.parent = temp
            temp.set_color(node_to_del.get_color())
        if orig_color == "black":
            self.delete_fix(cpy_node)

        self.size -= 1
    """
    Recolors nodes when the newly inserted node has a red parent and a red uncle.
    It's parameters are insrtd_node: Node, uncle: Node. 
    insrtd_node: The newly inserted node causing a red-red violation
    The parent and uncle become black, The grandparent becomes red.
    Moves up to the grandparent for further corrections
    
    """
    
    def _set_clor_swap(self: T, insrtd_node: Node, uncle: Node ) -> None:
        uncle.set_color("black")
        insrtd_node.parent.set_color("black")
        insrtd_node.parent.parent.set_color("red")
        insrtd_node = insrtd_node.parent.parent

    """
    Performs recoloring and rotation when the uncle node is black.
    Parameters:insrtd_node:newly inserted node which causes the imbalance, direction: direction of rotation
    """
    def _set_color_rotate(self: T, insrtd_node: Node, rot_dir: str)-> None:
         insrtd_node.parent.set_color("black")
         insrtd_node.parent.parent.set_color("red")
         self.rotate(insrtd_node.parent.parent,rot_dir)

    """
    Fix insert fixes the RebBlacktree after the insert if it violates the red black tree properties
    Parameters:Self: T - instance of the RebBlacktree
    Node: inserted node which causes the unbalance 
    It's return type is none
    
    """
    # Balance the tree after insertion
    def fix_insert(self: T, node: Node) -> None:
        while node.parent.is_red():
            if node.parent == node.parent.parent.right:
                uncle_node = node.parent.parent.left
                if uncle_node.is_red():
                    self._set_clor_swap(node, uncle_node)
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self.rotate(node,"right")
                    self._set_color_rotate(node, "left")

            else:
                uncle_node = node.parent.parent.right

                if uncle_node.is_red():
                    self._set_clor_swap(node, uncle_node)
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self.rotate(node,"left")

                    self._set_color_rotate(node, "right")
            
            # when node is root break
            if node == self.root:
                break
        
        #set color of root to black always
        self.root.set_color("black")

    def search(self: T, key: str) -> Node:
        return self._seach_rbt(self.root, key)

    def minimum(self: T, node: Node ) -> Node:
        if node.is_null():
            return self.terminal_null
        while not node.left.is_null():
            node = node.left
        return node




    def successor(self: T, node: Node) -> Node:
    # Case 1: If the node has a right child
        if not node.right.is_null():
            # Find the leftmost child in the right subtree
            succ_node = node.right
            while not succ_node.left.is_null():
                succ_node = succ_node.left
            return succ_node

        # Case 2: If the node does not have a right child
        succ_node = node.parent
        while succ_node is not None and node == succ_node.right:
            node = succ_node
            succ_node = succ_node.parent
        
        return succ_node


    

    def predecessor(self: T, node: Node) -> Node:
        if not node.left.is_null():
            pred_node = node.left
            while not pred_node.right.is_null():
                pred_node = pred_node.right
            return pred_node

        pred_node = node.parent
        while pred_node is not None and node == pred_node.left:
            node = pred_node
            pred_node = pred_node.parent

        return pred_node


    """
    Performs rotation (left or right) on the given pivot node in the Red-Black Tree to rebalance the tree
    Parameters:pivot (Node): The node around which the rotation is performed and direction left or right
    Riases exception when direction is not left or right
    """

    def rotate(self:T, pivot: Node, direction: str)-> None:
        if direction == "left":
            pivot_rchild = pivot.right
            pivot.right = pivot_rchild.left

            if not pivot_rchild.left.is_null():
                pivot_rchild.left.parent = pivot

            pivot_rchild.parent = pivot.parent

            if pivot.parent is None:
                self.root = pivot_rchild
            elif pivot == pivot.parent.left:
                pivot.parent.left = pivot_rchild
            else:
                pivot.parent.right = pivot_rchild

            pivot_rchild.left = pivot
            pivot.parent = pivot_rchild

        elif direction == "right":
            pivot_lchild = pivot.left
            pivot.left = pivot_lchild.right

            if not pivot_lchild.right.is_null():
                pivot_lchild.right.parent = pivot

            pivot_lchild.parent = pivot.parent
            if pivot.parent is None:
                self.root = pivot_lchild
            elif pivot == pivot.parent.right:
                pivot.parent.right = pivot_lchild
            else:
                pivot.parent.left = pivot_lchild
            pivot_lchild.right = pivot
            pivot.parent = pivot_lchild
        else:
            raise Exception("unknown direction for rotation")

    """
    initaliaze a new node 
    To reduce the complexity every node initiazed is taken as the red node
    """

    def create_rbt_init_node(self: T, key: str, value:int) -> Node:
        node = Node(key)
        node.left = self.terminal_null
        node.right = self.terminal_null
        node.set_color("red")
        node.set_value(value)

        return node

    """
    Returns the parent of the given node in the Red-Black Tree.
    It return the parent node, if there is no patent exist it returns none
    """
    
    def find_parent_for_insert(self: T, root_node: Node, new_node: Node) -> Node:
        parent_node = None
        current = root_node
        
        while not current.is_null():
            parent_node = current
            if new_node.get_key() < current.get_key():
                current = current.left
            else:
                current= current.right
        return parent_node

    def get_grand_parent(self: T, node: Node) -> Node:
        return node.parent.parent
    

    """
    inserting a new node to RBT while maintaing balance .
    Creates a node, find its parent and insert it , if there is any inbalnace craeted the fix insert will be called.
    IF the tree is emoty, the new node is inserted as a black node in RBT
    """
    def insert(self: T, key: str, value: int) -> bool:
        node = self.create_rbt_init_node(key, value)

        node.parent = self.find_parent_for_insert(self.root, node)
        

        par = node.parent

        if par is None:
            self.root = node
        elif node.get_key() < par.get_key():
            par.left = node
        else:
            par.right = node

        self.size += 1

        if node.is_parent_null():
            node.set_color("black")
            return True

        if self.get_grand_parent(node) is None:
            return True

        self.fix_insert(node)
        
        return True


    def delete(self: T, key: str) -> None:
        self._find_and_delete(self.root, key)
    
    

    

  

class plateMgmt:
    def __init__(self: T, output_file: str)-> None:
        self.tree = RedBlackTree()
        self.ouput_file =  output_file

    # generates the random number plate with length 4, combiantion of digoits and uppercase character
    def _genUniqePlateVal(self: T) -> str:
        length = 4
        characters = string.ascii_uppercase + string.digits 
        plateVal = ''.join(random.choice(characters) for _ in range(length))


        while not self.tree.search(plateVal).is_null() :
            plateVal = ''.join(random.choice(characters) for _ in range(length))
        #print(plateVal)
        return plateVal
    
    def writeToOutputFile(self: T, msg : str) -> None:
        with open(self.ouput_file, 'a', encoding="utf-8") as f:
            print(msg, file=f)
    
    # create_and_addLicence is used to generate a random number plate 
    # it cost =4
    # after craetion of the licence , it get inserted into the  tree

    def create_and_addLicence(self: T) -> bool:
        plate_num = self._genUniqePlateVal()
        plate_fee = 4 # 4 Galleons annually + 3 Galleons extra for custom plate

        if self.tree.size == 0:
            self.tree.insert(plate_num, plate_fee)
            output_string = plate_num + " created and registered successfully." 
            self.writeToOutputFile(output_string)
            #print(plate_num + " created and registered successfully.")
            return True
        
        if self.tree.size > 0:
            self.tree.insert(plate_num, plate_fee)
            output_string = plate_num + " created and registered successfully."
            self.writeToOutputFile(output_string)
            #print(plate_num + " created and registered successfully.")
            return True
        
        return False
    
    #addLicence is used to add custom or random license plate based on the key provided
    # if key is not privided will create a random number plate
    # if key is provided it will check agianst the tree and if the key doesn't exist , the function will insert the new custom number plate

    def addLicence(self: T, key: str = None)-> bool:

        if key == None: # key is not provided which means we have to create a key ourseves. This is not a custom key
            self.create_and_addLicence()
            return True
        
        #key is provided so it is a custom key
        custom_key_total_fees = 7 # 4 Galleons annually + 3 Galleons extra for custom plate
        if self.tree.size == 0:
            self.tree.insert(key, custom_key_total_fees)
            output_string = key + " registered successfully."
            self.writeToOutputFile(output_string)
            #print (key + " registered successfully.")
            return True
        
        if self.tree.size > 0:
            if self.tree.search(key).is_null() :
                self.tree.insert (key, custom_key_total_fees) 
                output_string = key +" registered successfully."
                self.writeToOutputFile(output_string)
                #print (key + " registered successfully.")
                return True
            else :
                output_string = "Failed to register " + key + ": already exists." 
                self.writeToOutputFile(output_string)
                #print("Failed to register " + key + ": already exists.)
            
                return True
            
    # dropLicence is the function used to delete certain license plate.
    # if the key exist in the tree it will delete the key and balance the tree
    # if the doesn't exist in the tree it will retuurn key doesn't exist

    def dropLicence(self: T, key: str) -> None:

        if self.tree.size == 0:
            output_string = "Failed to remove " + key + ": does not exist."
            self.writeToOutputFile(output_string)
            #print("Failed to remove " + key + ": does not exist.")
            return
        
        if self.tree.search(key).is_null() :
            output_string ="Failed to remove " + key + ": does not exist." 
            self.writeToOutputFile(output_string)
            #print("Failed to remove " + key +": does not exist.")
            return
        else :
            self. tree.delete(key)
            output_string = key + " removed successfully."
            self.writeToOutputFile(output_string)
            #print( key + " removed successfully." )




    def lookupLicence(self: T, key: str) -> None:

        if self.tree.size == 0:
            output_string = key + " does not exist."
            self.writeToOutputFile(output_string)
            #print(key + " does not exist.")
            return
        
        if self.tree.search(key).is_null() :
            output_string = key + " does not exist." 
            self.writeToOutputFile(output_string)
            #print(key + " does not exist.")
            return
        else :
            output_string = key + " exists."
            self.writeToOutputFile(output_string)
            #print(key + "exists.")

    # lookupPrev returns the predessor

    def lookupPrev(self: T, key: str) -> None:

        if self.tree.size == 0:
            output_string ="No elements in the tree cannot find prev for key "+ key 
            self.writeToOutputFile(output_string)
            #print("No elements in the tree cannot find prev for key "+ key)
            return
        
        if self.tree.search(key).is_null() :
            output_string = key + " does not exist cannot find predessor"
            self.writeToOutputFile(output_string)
            #print(key + " does not exist cannot find predessor") 
            return
        else :
            prev = self.tree.predecessor(self.tree.search(key))
            if prev == None:
                output_string = key + "'s prev is does not exist it is the first node."
                self.writeToOutputFile(output_string)
                return

            output_string = key + "'s prev is "+ prev._key + "."
            self.writeToOutputFile(output_string)
            #print( key + "'s prev is "+ prev._key + ".")

    # lookupNext return the successor if there exist any
    def lookupNext(self: T, key: str) -> None:
        if self.tree.size == 0:
            output_string = "No elements in the tree cannot find next for key "+ key + "."
            self.writeToOutputFile(output_string)
            #print ("No elements in the tree cannot find next for key "+ key)
            return
        

        if self.tree.search(key).is_null() :
            output_string = key +" does not exist cannot find prev."
            self.writeToOutputFile(output_string)
            #print(key + " does not exist cannot find prev")
            return
        else :
            next = self.tree.successor(self.tree.search(key)) 
            if next == None:
                output_string = key + "'s next does not exist it is the last node."
                self.writeToOutputFile(output_string)
                return

            output_string = key + "'s next is "+ next._key + "."
            self.writeToOutputFile(output_string)
            #print( key + "'s next is "+ next._key + ".")
    
    #lookupRange is used to seacrh and print all the keys between lo and hi keys. 
        
    def lookupRange(self: T, lo: str, hi: str) -> None:
        inorder_nodes = self.tree.inorder()
        output_string = 'Plate numbers between ' + lo + ' and ' + hi +": " 
        result =''

        for node in inorder_nodes:
            if lo <= node._key <=hi:
                if len(result) == 0:
                    result += node._key 
                else:
                    result +=", " + node._key
                
        result = result + "."
        output_string = output_string + result
        self.writeToOutputFile(output_string)


    def revenue (self: T) -> None:
        inorder_nodes = self. tree.inorder()
        total_revenue = 0 
        
        for node in inorder_nodes:
            total_revenue += node.value

    
        output_string = "Current annual revenue is "
        #total revenue will be a number so should convert to 
        # #string for writing to file because writeToOutputFile accepts a string 
        output_string = output_string + str(total_revenue) + " Galleons."
        self.writeToOutputFile(output_string)


    def quit(self: T)-> None:
        sys.exit()


class iOManager:
    def __init__(self: T,fileName:str) -> None:
        self.input_filename = fileName
        f_name = os.path.splitext(fileName)[0]
        self.out_filename = f_name + "_" + "output.txt"
        self.function_names = ['addLicence', 'dropLicence', 'lookupRange','lookupPrev', 'lookupNext', 'lookupLicence','revenue','quit']

    def ReadInput(self: T):
        with open(self.input_filename, 'r') as file:
            lines = file.readlines()

        functions_data = []

        for line in lines:
            line = line. strip()

            for name in self. function_names: 
                if line. startswith(name):
                        start_par = line.find('(')
                        end_par = line.find(')')

                        if start_par != -1 and end_par != -1:
                            args_str = line[start_par+1 : end_par]
                            args = args_str.split(',')
                            args = [arg.strip() for arg in args]
                            # fix this condition
                            args = [arg for arg in args if len(arg) == 4 and arg.isalnum () ]
                            
                            # for arg in args:
                            #   if len(arg) == 4 and arg.isalnum() :
                            functions_data.append ((name, args))
        return functions_data

if __name__ == '__main__':
    fileName = sys.argv[1]
    io = iOManager(fileName)
    operationList = io.ReadInput()
    pt = plateMgmt(io.out_filename)

    for operation in operationList:
        function_name = operation [0]
        args = operation[1]

        if function_name == pt.addLicence.__name__:
            if len(args) == 1:
                pt.addLicence(args[0])
            elif len(args) == 0:
                pt.addLicence()
        elif function_name == pt.dropLicence.__name__:
                pt.dropLicence(args[0])
        elif function_name == pt.lookupLicence.__name__:
                pt.lookupLicence(args[0])
        elif function_name == pt.lookupPrev.__name__:
                pt.lookupPrev(args[0])
        elif function_name == pt.lookupNext.__name__:
                pt.lookupNext(args[0])
        elif function_name == pt.lookupRange.__name__:
                pt.lookupRange(args[0],args[1])
        elif function_name == pt.revenue.__name__:
                pt.revenue()
        elif function_name == pt.quit.__name__:
                pt.quit()
    
