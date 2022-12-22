exec(open('imports.py').read())

from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles

fig,ax=plt.subplots(figsize=(6,8))
#ax = fig.add_subplot(121)
v1= venn2( subsets=(5, 0, 5), set_labels = ('Group A', 'Group B'), alpha = 1.0, set_colors=('blue','red'))
venn2_circles(subsets=(10, 0, 10))
for text in v1.set_labels:
    text.set_fontsize(14)
for text in v1.subset_labels:
    text.set_fontsize(16)
# Hide the labels both set_labels and subset_labels
for idx, subset in enumerate(v1.set_labels):
    v1.set_labels[idx].set_visible(False)
# Change Backgroud
#plt.gca().set_axis_bgcolor('blue')
plt.gca().set_axis_on()
v1.get_label_by_id('10').set_text('A')
v1.get_label_by_id('11').set_text('B')
v1.get_label_by_id('01').set_text('')

 
#ax = fig.add_subplot(122)
#venn3(subsets = (20, 10, 12, 10, 9, 4, 3), set_labels = ('Group A', 'Group B', 'Group C'), alpha = 0.5)

plt.savefig('venn.pdf')

# Left panel =========================================
fig,ax=plt.subplots(figsize=(6,6))
v3=venn3(subsets = (50, 00, 15, 0, 10, 0, 5), set_labels = ('Group A', 'Group B', 'Group C'), alpha = 1.0, set_colors=('none','red','yellow'))
v3.get_label_by_id('100').set_text('$\Omega$')
v3.get_label_by_id('110').set_text('A')
v3.get_patch_by_id('100').set_alpha(0.0) # remove this color
v3.get_label_by_id('100').set_y(0.4)
v3.get_label_by_id('101').set_text('B')
v3.get_label_by_id('111').set_text('A$\cap$B')
for idx, subset in enumerate(v3.set_labels):
    v3.set_labels[idx].set_visible(False)
# add circles around the sets
c = venn3_circles(subsets =  (50, 0, 15, 0, 10, 0, 5), linestyle='solid', linewidth=2, color='black')
plt.savefig('venn1.pdf')

# Right panel ========================================
ig,ax=plt.subplots(figsize=(6,6))
v3=venn3(subsets = (50, 0, 15, 0, 10, 0, 0), set_labels = ('Group A', 'Group B', 'Group C'), alpha = 1.0, set_colors=('none','red','yellow'))
v3.get_label_by_id('100').set_text('$\Omega$')
v3.get_label_by_id('110').set_text('A')
v3.get_patch_by_id('100').set_alpha(0.0) # remove this color
v3.get_label_by_id('100').set_y(0.4)
v3.get_label_by_id('101').set_text('B')
#v3.get_label_by_id('111').set_text('A$\cap$B')
for idx, subset in enumerate(v3.set_labels):
    v3.set_labels[idx].set_visible(False)
# add circles around the sets
c = venn3_circles(subsets =  (50, 00, 15, 0, 10, 0, 0), linestyle='solid', linewidth=2, color='black')
plt.savefig('venn2.pdf')




