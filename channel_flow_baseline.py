import os
import warnings

import numpy as np
from sympy import Symbol, Eq, And, Or

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.utils.io import (csv_to_dict, ValidatorPlotter, InferencerPlotter,)
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from modulus.sym.utils.sympy.functions import parabola
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.node import Node

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # add constraints to solver
    # simulation params
    channel_length = 10
    channel_width = 2
    chip_height_percent = 0.5
    chip_height = channel_width * chip_height_percent
    chip_width = 1
    chip_pos = (channel_length - 5 * chip_width)/4
    inlet_vel = 0.1

    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")

        # define geometry
    channel = Channel2D(
        (0, 0), (channel_length, channel_width)
    )
    inlet = Line(
        (0, chip_height),
        (0, channel_width),
        normal=1,
    )
    outlet = Line(
        (channel_length, chip_height),
        (channel_length, channel_width),
        normal=1,
    )
    rec1 = Rectangle(
        (0, 0),
        (chip_width, chip_height),
    )
    rec2 = Rectangle(
        (chip_pos + chip_width, channel_width-chip_height),
        (chip_pos + 2*chip_width, channel_width),
    )
    rec3 = Rectangle(
        (2*chip_pos + 2*chip_width, 0),
        (2*chip_pos + 3*chip_width, chip_height),
    )
    rec4 = Rectangle(
        (3*chip_pos + 3*chip_width, channel_width-chip_height),
        (3*chip_pos + 4*chip_width, channel_width),
    )
    rec5 = Rectangle(
        (4*chip_pos + 4*chip_width, 0),
        (4*chip_pos + 5*chip_width, chip_height),
    )
    geo = channel - rec1 - rec2 - rec3 - rec4 -rec5
    
    x_pos = Symbol("x_pos")
    integral_line = Line((x_pos, 0), (x_pos, channel_width), 1)
    x_pos_range = {
        x_pos: lambda batch_size: np.full(
            (batch_size, 1), np.random.uniform(0, channel_length)
        )
    }

    # make domain
    domain = Domain()

    # inlet
    #inlet_parabola = parabola(y, chip_height, channel_width, inlet_vel)
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"normal_dot_vel":inlet_vel},
        #lambda_weighting={"p": 10},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        #lambda_weighting={"p": 10},
        batch_size=cfg.batch_size.outlet,
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    # # interior lr
    # interior_lr = PointwiseInteriorConstraint(
    #     nodes=nodes,
    #     geometry=geo,
    #     outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
    #     batch_size=cfg.batch_size.interior_lr,
    #     criteria=Or(x < (chip_pos - 0.25), x > (chip_pos + chip_width + 0.25)),
    #     lambda_weighting={
    #         "continuity": 2 * Symbol("sdf"),
    #         "momentum_x": 2 * Symbol("sdf"),
    #         "momentum_y": 2 * Symbol("sdf"),
    #     },
    # )
    # domain.add_constraint(interior_lr, "interior_lr")

    # interior hr
    interior_hr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_hr,
        # criteria=And(x > (chip_pos - 0.25), x < (chip_pos + chip_width + 0.25)),
        lambda_weighting={
            "continuity": 2 * Symbol("sdf"),
            "momentum_x": 2 * Symbol("sdf"),
            "momentum_y": 2 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior_hr, "interior_hr")

    # integral continuity
    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel":inlet_vel},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 1},
        criteria=integral_criteria,
        parameterization=x_pos_range,
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    # add validation data
    file_path = "comsol/comsol_channel_flow_baseline.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"Points:0": "x", "Points:1": "y", "Velocity_field,_x_component": "u", "Velocity_field,_y_component": "v", "Pressure": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["x"] -= 0  # normalize pos
        openfoam_var["y"] -= 0
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["u", "v", "p"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            batch_size=1024,
            plotter=ValidatorPlotter(),
        )
        domain.add_validator(openfoam_validator)

        # add inferencer data
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            output_names=["u", "v", "p"],
            batch_size=1024,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, "inf_data")
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_sym_examples_supplemental_materials"
        )

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()