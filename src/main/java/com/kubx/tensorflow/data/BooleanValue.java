package com.kubx.tensorflow.data;

import java.nio.ByteBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class BooleanValue extends Value<BooleanValue, ByteBuffer> {
 
  public static BooleanValue of(Operand<Boolean> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static BooleanValue of(Output<Boolean> output) {
    return of((Tensor<Boolean>)output.tensor());
  }

  public static BooleanValue of(Tensor<Boolean> t) {
    return of(t.buffer(), t.shape());
  }

  public static BooleanValue of(ByteBuffer buffer, long[] shape) {
    return new BooleanValue(buffer, 0, toIndices(shape));
  }
  
  public boolean scalar() {
    if (indices.size() > 0) {
      throw new IllegalArgumentException("Cannot convert value of " + indices.size() + " dimensions to scalar");
    }
    return buffer.get(0) > 0;
  }
  
  @Override
  protected BooleanValue newValue(ByteBuffer buffer, int position, List<Index> indices) {
    return new BooleanValue(buffer, position, indices);
  }
  
  @Override
  protected ByteBuffer slice() {
    return buffer.slice();
  }
  
  private BooleanValue(ByteBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
