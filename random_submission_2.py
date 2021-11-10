from zerosum_env.envs.carbon.helpers import RecrtCenterAction, WorkerAction, Board
from random import choice, randint, sample
def agent(observation, configuration):
    board = Board(observation, configuration)
    me = board.current_player
    remaining_carbon = me.cash
    workers = me.workers
    # randomize worker order
    workers = sample(workers, len(workers))
    actions = {}
    for worker in workers:
        if worker.cell.carbon > worker.carbon and randint(0, 1) == 0:
            # 50% chance to mine
            continue
        if worker.cell.recrtCenter is None and remaining_carbon > board.configuration.plant_cost:
            # 5% chance to convert at any time
            if randint(0, 19) == 0:
                continue
            # 50% chance to convert if there are no recrtCenters
            if randint(0, 1) == 0 and len(me.recrtCenters) == 0:
                continue
        # None represents the chance to do nothing
        worker.next_action = choice(WorkerAction.moves())
    recrtCenters = me.recrtCenters
    # randomize recrtCenter order
    recrtCenters = sample(recrtCenters, len(recrtCenters))
    worker_count = len(board.next().current_player.workers)
    for recrtCenter in recrtCenters:
        # If there are no workers, always spawn if possible
        if worker_count == 0 and remaining_carbon > board.configuration.rec_collector_cost:
            remaining_carbon -= board.configuration.rec_collector_cost
            recrtCenter.next_action = RecrtCenterAction.RECCOLLECTOR
            
        # 20% chance to spawn if no workers
        elif randint(0, 4) == 0 and remaining_carbon > board.configuration.rec_planter_cost:
            remaining_carbon -= board.configuration.rec_planter_cost
            recrtCenter.next_action = RecrtCenterAction.RECPLANTER
    # print(me.next_actions)
    return me.next_actions

    