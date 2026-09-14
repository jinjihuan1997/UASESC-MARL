from diag_support import *
os.environ['MPLCONFIGDIR']=str(HERE/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 guard();m=manifest();fig,axs=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
 for ax,seed in zip(axs,m['parents']):
  for step,color in [(1000000,'#356cac'),(6000000,'#d37520')]:
   z=arrays(HERE/f'analysis_arrays/{seed}_{step}_fixed_2.npz');x=np.sort(z['deltaQ'].ravel());ax.plot(x,np.arange(1,len(x)+1)/len(x),label=f'{step//1000000}M steps',color=color,lw=2)
  ax.axvline(0,color='#555',lw=1);ax.axvline(.005,color='#aa3333',ls='--',lw=1,label='Quality threshold');ax.set_title(str(seed));ax.set_xlabel('Paired quality gain vs own deterministic policy');ax.grid(alpha=.2);ax.set_ylim(0,1.02)
 axs[0].set_ylabel('Empirical cumulative probability');axs[-1].legend(fontsize=8,loc='lower right');fig.suptitle('Fixed Quality: same environment, different sampled actions',fontsize=13);fig.tight_layout();fig.savefig(HERE/'report/quality_tail_cdf.png',dpi=160);plt.close(fig)
 g=read(HERE/'report/gradient_results.json');labels=['expected_quality','negative_AoI_cost','negative_resource_cost'];v=np.array([[g[str(s)]['joint']['cosine']['tail_quality'][k] for k in labels] for s in m['parents']]);fig,ax=plt.subplots(figsize=(7.5,4));im=ax.imshow(v,cmap='RdBu',vmin=-1,vmax=1);ax.set_xticks(range(3),['Increase\nquality','Reduce\nAoI cost','Reduce\nresource cost'],fontsize=10);ax.set_yticks(range(3),list(map(str,m['parents'])));ax.set_title('TailRL(Q) gradient direction: cosine with each objective',fontsize=11)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{v[i,j]:.3f}',ha='center',va='center',color='white' if abs(v[i,j])>.7 else 'black')
 fig.colorbar(im,ax=ax,shrink=.8);fig.tight_layout();fig.savefig(HERE/'report/gradient_directions.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
