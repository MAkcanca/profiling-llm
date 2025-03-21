###############################################################################
#
#  Welcome to Baml! To use this generated code, please run the following:
#
#  $ pip install baml-py
#
###############################################################################

# This file was generated by BAML: please do not edit it. Instead, edit the
# BAML files and re-generate this code.
#
# ruff: noqa: E501,F401
# flake8: noqa: E501,F401
# pylint: disable=unused-import,line-too-long
# fmt: off
from typing import Any, Dict, List, Optional, TypeVar, Union, TypedDict, Type, Literal, cast
from typing_extensions import NotRequired
import pprint

import baml_py
from pydantic import BaseModel, ValidationError, create_model

from . import partial_types, types
from .types import Checked, Check
from .type_builder import TypeBuilder
from .globals import DO_NOT_USE_DIRECTLY_UNLESS_YOU_KNOW_WHAT_YOURE_DOING_CTX, DO_NOT_USE_DIRECTLY_UNLESS_YOU_KNOW_WHAT_YOURE_DOING_RUNTIME

OutputType = TypeVar('OutputType')

# Define the TypedDict with optional parameters having default values
class BamlCallOptions(TypedDict, total=False):
    tb: NotRequired[TypeBuilder]
    client_registry: NotRequired[baml_py.baml_py.ClientRegistry]
    collector: NotRequired[Union[baml_py.baml_py.Collector, List[baml_py.baml_py.Collector]]]

class BamlSyncClient:
    __runtime: baml_py.BamlRuntime
    __ctx_manager: baml_py.BamlCtxManager
    __stream_client: "BamlStreamClient"

    def __init__(self, runtime: baml_py.BamlRuntime, ctx_manager: baml_py.BamlCtxManager):
      self.__runtime = runtime
      self.__ctx_manager = ctx_manager
      self.__stream_client = BamlStreamClient(self.__runtime, self.__ctx_manager)

    @property
    def stream(self):
      return self.__stream_client

    
    def EvaluatePaper(
        self,
        paper_content: str,
        baml_options: BamlCallOptions = {},
    ) -> str:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)
      collector = baml_options.get("collector", None)
      collectors = collector if isinstance(collector, list) else [collector] if collector is not None else []

      raw = self.__runtime.call_function_sync(
        "EvaluatePaper",
        {
          "paper_content": paper_content,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        collectors,
      )
      return cast(str, raw.cast_to(types, types, partial_types, False))
    
    def ExtractProfile(
        self,
        case_data: str,
        baml_options: BamlCallOptions = {},
    ) -> types.ProfileResult:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)
      collector = baml_options.get("collector", None)
      collectors = collector if isinstance(collector, list) else [collector] if collector is not None else []

      raw = self.__runtime.call_function_sync(
        "ExtractProfile",
        {
          "case_data": case_data,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        collectors,
      )
      return cast(types.ProfileResult, raw.cast_to(types, types, partial_types, False))
    



class BamlStreamClient:
    __runtime: baml_py.BamlRuntime
    __ctx_manager: baml_py.BamlCtxManager

    def __init__(self, runtime: baml_py.BamlRuntime, ctx_manager: baml_py.BamlCtxManager):
      self.__runtime = runtime
      self.__ctx_manager = ctx_manager

    
    def EvaluatePaper(
        self,
        paper_content: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.BamlSyncStream[Optional[str], str]:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)
      collector = baml_options.get("collector", None)
      collectors = collector if isinstance(collector, list) else [collector] if collector is not None else []

      raw = self.__runtime.stream_function_sync(
        "EvaluatePaper",
        {
          "paper_content": paper_content,
        },
        None,
        self.__ctx_manager.get(),
        tb,
        __cr__,
        collectors,
      )

      return baml_py.BamlSyncStream[Optional[str], str](
        raw,
        lambda x: cast(Optional[str], x.cast_to(types, types, partial_types, True)),
        lambda x: cast(str, x.cast_to(types, types, partial_types, False)),
        self.__ctx_manager.get(),
      )
    
    def ExtractProfile(
        self,
        case_data: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.BamlSyncStream[partial_types.ProfileResult, types.ProfileResult]:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)
      collector = baml_options.get("collector", None)
      collectors = collector if isinstance(collector, list) else [collector] if collector is not None else []

      raw = self.__runtime.stream_function_sync(
        "ExtractProfile",
        {
          "case_data": case_data,
        },
        None,
        self.__ctx_manager.get(),
        tb,
        __cr__,
        collectors,
      )

      return baml_py.BamlSyncStream[partial_types.ProfileResult, types.ProfileResult](
        raw,
        lambda x: cast(partial_types.ProfileResult, x.cast_to(types, types, partial_types, True)),
        lambda x: cast(types.ProfileResult, x.cast_to(types, types, partial_types, False)),
        self.__ctx_manager.get(),
      )
    

b = BamlSyncClient(DO_NOT_USE_DIRECTLY_UNLESS_YOU_KNOW_WHAT_YOURE_DOING_RUNTIME, DO_NOT_USE_DIRECTLY_UNLESS_YOU_KNOW_WHAT_YOURE_DOING_CTX)

__all__ = ["b"]