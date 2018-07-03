//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <memory>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/platform/place.h"

#include <memory>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {

class ReaderBase : public std::enable_shared_from_this<ReaderBase> {
 public:
  virtual void ReadNext(std::vector<LoDTensor>* out) = 0;

  virtual void ReInit() = 0;

  void Decorate(const std::shared_ptr<ReaderBase>& reader) {
    decorated_readers_.emplace_back(reader);
  }

  virtual std::function<void()> CloseAndOpen() {
    // Close()
    return [] {};
  }

  void ResetAll() {
    std::vector<std::function<void()>> open_methods;
    std::unordered_set<ReaderBase*> visited;

    for (auto& dreader : decorated_readers_) {
      TraceAllDecoratedReader(dreader, open_methods, visited);
    }
    open_methods.emplace_back(CloseAndOpen());

    for (auto it = open_methods.rbegin(); it != open_methods.rend(); ++it) {
      (*it)();
    }
  }

  static void TraceAllDecoratedReader(
      std::weak_ptr<ReaderBase> reader,
      std::vector<std::function<void()>>& open_methods,
      std::unordered_set<ReaderBase*> visited) {
    if (auto real_reader = reader.lock()) {
      auto* ptr = real_reader.get();
      if (visited.count(ptr)) {
        return;
      }

      for (auto& dreader : ptr->decorated_readers_) {
        TraceAllDecoratedReader(dreader, open_methods, visited);
      }
      open_methods.emplace_back(ptr->CloseAndOpen());
    }
  }

 protected:
  std::vector<std::weak_ptr<ReaderBase>> decorated_readers_;
};

class DecoratedReader : public ReaderBase {
 public:
  explicit DecoratedReader(const std::shared_ptr<ReaderBase>& reader)
      : ReaderBase(), reader_(reader) {
    reader_->Decorate(shared_from_this());
    PADDLE_ENFORCE_NOT_NULL(reader_);
  }

  void ReInit() override { reader_->ReInit(); }

  std::function<void()> CloseAndOpen() final {
    auto open_method = this->CloseAndOpenMyself();
    auto underlying_open_method = reader_->CloseAndOpen();

    return [=] {
      underlying_open_method();
      open_method();
    };
  }

 protected:
  virtual std::function<void()> CloseAndOpenMyself() {
    return [] {};
  }

  std::shared_ptr<ReaderBase> reader_;
};

class FileReader : public ReaderBase {
 public:
  explicit FileReader(const std::vector<DDim>& dims);

  void ReadNext(std::vector<LoDTensor>* out) override;

 protected:
  virtual void ReadNextImpl(std::vector<LoDTensor>* out) = 0;

 private:
  std::vector<DDim> dims_;
};

// The ReaderHolder is used as reader' unified wrapper,
// making it easier to access different type reader in Variables.
class ReaderHolder {
 public:
  void Reset(ReaderBase* reader) { reader_.reset(reader); }

  std::shared_ptr<ReaderBase> Get() const { return reader_; }

  void ReadNext(std::vector<LoDTensor>* out) {
    PADDLE_ENFORCE_NOT_NULL(reader_);
    reader_->ReadNext(out);
  }
  void ReInit() {
    PADDLE_ENFORCE_NOT_NULL(reader_);
    reader_->ReInit();
  }

 private:
  std::shared_ptr<ReaderBase> reader_;
};

}  // namespace framework
}  // namespace paddle
