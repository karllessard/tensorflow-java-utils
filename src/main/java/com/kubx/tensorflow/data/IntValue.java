package com.kubx.tensorflow.data;

import java.nio.IntBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class IntValue extends Value<IntValue, IntBuffer> {
 
  public static IntValue of(Operand<Integer> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static IntValue of(Output<Integer> output) {
    return of((Tensor<Integer>)output.tensor());
  }

  public static IntValue of(Tensor<Integer> t) {
    return of(t.buffer().asIntBuffer(), t.shape());
  }

  public static IntValue of(IntBuffer buffer, long[] shape) {
    return new IntValue(buffer, 0, toIndices(shape));
  }
  
  public int scalar() {
    if (indices.size() > 0) {
      throw new IllegalArgumentException("Cannot convert value of " + indices.size() + " dimensions to scalar");
    }
    return buffer.get(position);
  }
  
  @Override
  protected IntValue newValue(IntBuffer buffer, int position, List<Index> indices) {
    return new IntValue(buffer, position, indices);
  }
  
  @Override
  protected IntBuffer slice() {
    return buffer.slice();
  }
  
  private IntValue(IntBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
