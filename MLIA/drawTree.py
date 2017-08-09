'''
1. Collect: Text file provided
2. Prepare: Parse tab-delimited lines
3. Analyze: Plot final tree with createPlot()
4. Train: Use createTree()
5. Test: Write a function to descend the tree for a given instance
6. Use: Persist the tree data structure so it can be recalled without
building the tree
'''
import trees
import treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
treePlotter.createPlot(lensesTree)