package com.kubx.tensorflow.data;

import java.nio.LongBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class LongValue extends Value<LongValue, LongBuffer> {
 
  public static LongValue of(Operand<Long> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static LongValue of(Output<Long> output) {
    return of((Tensor<Long>)output.tensor());
  }

  public static LongValue of(Tensor<Long> t) {
    return of(t.buffer().asLongBuffer(), t.shape());
  }

  public static LongValue of(LongBuffer buffer, long[] shape) {
    return new LongValue(buffer, 0, toIndices(shape));
  }
  
  public long scalar() {
    if (indices.size() > 0) {
      throw new IllegalArgumentException("Cannot convert value of " + indices.size() + " dimensions to scalar");
    }
    return buffer.get(0);
  }
  
  @Override
  protected LongValue newValue(LongBuffer buffer, int position, List<Index> indices) {
    return new LongValue(buffer, position, indices);
  }
  
  @Override
  protected LongBuffer slice() {
    return buffer.slice();
  }
  
  private LongValue(LongBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
