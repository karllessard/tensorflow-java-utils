package com.kubx.tensorflow.data;

public abstract class Index {
  
  protected static class Combined extends Index {

    @Override
    public int size() {
      return frontIndex.size();
    }

    @Override
    protected int position(int i) {
      return backIndex.position(frontIndex.position(i));
    }

    @Override
    protected boolean vectorizable() {
      return backIndex.vectorizable() && frontIndex.vectorizable();
    }

    protected Combined(Index backIndex, Index frontIndex) {
      this.backIndex = backIndex;
      this.frontIndex = frontIndex;
    }
    
    private Index backIndex;
    private Index frontIndex;
  }
  
  public abstract int size();
  
  protected abstract int position(int i);
  
  protected boolean vectorizable() {
    return false;
  }
  
  protected Index merge(Index previousIndex) {
    return new Combined(previousIndex, this);
  }
}
