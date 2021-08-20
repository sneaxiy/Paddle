// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/cumprod_op.h"

namespace paddle {
namespace operators {

class CumprodOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->ShareDim("X", "Out");
    ctx->ShareLoD("X", "Out");
  }
};

class CumprodOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "");
    AddOutput("Out", "");
    AddAttr<int>("dim", "");
    AddComment(R"DOC(Cumprod op)DOC");
  }
};

template <typename T>
class CumprodGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("cumprod_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Out", this->Output("Out"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class CumprodGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->ShareDim(framework::GradVarName("Out"), framework::GradVarName("X"));
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(cumprod, ops::CumprodOp, ops::CumprodOpMaker,
                  ops::CumprodGradOpMaker<paddle::framework::OpDesc>,
                  ops::CumprodGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(cumprod_grad, ops::CumprodGradOp);

REGISTER_OP_CPU_KERNEL(
    cumprod, ops::CumprodOpCPUKernel<float>, ops::CumprodOpCPUKernel<double>,
    ops::CumprodOpCPUKernel<int>, ops::CumprodOpCPUKernel<int64_t>,
    ops::CumprodOpCPUKernel<paddle::platform::complex<float>>,
    ops::CumprodOpCPUKernel<paddle::platform::complex<double>>);

REGISTER_OP_CPU_KERNEL(
    cumprod_grad, ops::CumprodGradOpCPUKernel<float>,
    ops::CumprodGradOpCPUKernel<double>, ops::CumprodGradOpCPUKernel<int>,
    ops::CumprodGradOpCPUKernel<int64_t>,
    ops::CumprodGradOpCPUKernel<paddle::platform::complex<float>>,
    ops::CumprodGradOpCPUKernel<paddle::platform::complex<double>>);
