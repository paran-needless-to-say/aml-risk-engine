"""
이상행동 탐지를 위한 고급 피처 추출 모듈

fanout, hop, amount variance 등 복잡한 그래프 기반 피처들을 계산합니다.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import networkx as nx
from datetime import datetime, timedelta


def extract_advanced_features(transactions: List[Dict[str, Any]], 
                            address: str,
                            window_hours: int = 24) -> Dict[str, float]:
    """
    고급 피처들을 추출합니다.
    
    Args:
        transactions: 트랜잭션 리스트
        address: 분석할 주소
        window_hours: 분석 시간 윈도우
    
    Returns:
        고급 피처 딕셔너리
    """
    features = {}
    
    # 그래프 구성
    graph = _build_transaction_graph(transactions)
    
    # 그래프 기반 피처
    graph_features = _extract_graph_features(graph, address)
    features.update(graph_features)
    
    # 시간 패턴 피처
    temporal_features = _extract_temporal_features(transactions, address, window_hours)
    features.update(temporal_features)
    
    # 금액 패턴 피처
    amount_features = _extract_amount_features(transactions, address)
    features.update(amount_features)
    
    # 가스 패턴 피처
    gas_features = _extract_gas_features(transactions, address)
    features.update(gas_features)
    
    # 주소 패턴 피처
    address_features = _extract_address_features(transactions, address)
    features.update(address_features)
    
    return features


def _build_transaction_graph(transactions: List[Dict[str, Any]]) -> nx.DiGraph:
    """트랜잭션들로부터 방향 그래프를 구성합니다."""
    graph = nx.DiGraph()
    
    for tx in transactions:
        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()
        value = float(tx.get('value', 0))
        
        if from_addr and to_addr:
            if graph.has_edge(from_addr, to_addr):
                # 기존 엣지 가중치 업데이트
                graph[from_addr][to_addr]['weight'] += value
                graph[from_addr][to_addr]['count'] += 1
            else:
                # 새 엣지 추가
                graph.add_edge(from_addr, to_addr, weight=value, count=1)
    
    return graph


def _extract_graph_features(graph: nx.DiGraph, address: str) -> Dict[str, float]:
    """그래프 기반 피처들을 추출합니다."""
    features = {}
    address = address.lower()
    
    if address not in graph:
        return {
            'fanout_degree': 0.0,
            'fanin_degree': 0.0,
            'hop_2_neighbors': 0.0,
            'hop_3_neighbors': 0.0,
            'clustering_coefficient': 0.0,
            'betweenness_centrality': 0.0,
            'pagerank': 0.0,
            'eigenvector_centrality': 0.0
        }
    
    # Fanout/Fanin (출차수/입차수)
    features['fanout_degree'] = float(graph.out_degree(address))
    features['fanin_degree'] = float(graph.in_degree(address))
    
    # N-hop 이웃 수
    features['hop_2_neighbors'] = float(len(_get_n_hop_neighbors(graph, address, 2)))
    features['hop_3_neighbors'] = float(len(_get_n_hop_neighbors(graph, address, 3)))
    
    # 클러스터링 계수 (무방향 그래프로 변환)
    undirected_graph = graph.to_undirected()
    if undirected_graph.number_of_nodes() > 1:
        features['clustering_coefficient'] = nx.clustering(undirected_graph, address)
    else:
        features['clustering_coefficient'] = 0.0
    
    # 중심성 지표들
    try:
        # 매개 중심성
        betweenness = nx.betweenness_centrality(graph)
        features['betweenness_centrality'] = betweenness.get(address, 0.0)
        
        # PageRank
        pagerank = nx.pagerank(graph)
        features['pagerank'] = pagerank.get(address, 0.0)
        
        # 고유벡터 중심성
        try:
            eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
            features['eigenvector_centrality'] = eigenvector.get(address, 0.0)
        except:
            features['eigenvector_centrality'] = 0.0
            
    except:
        features['betweenness_centrality'] = 0.0
        features['pagerank'] = 0.0
        features['eigenvector_centrality'] = 0.0
    
    return features


def _get_n_hop_neighbors(graph: nx.DiGraph, node: str, n: int) -> Set[str]:
    """N-hop 이웃들을 찾습니다."""
    if n <= 0 or node not in graph:
        return set()
    
    visited = set()
    current_level = {node}
    
    for hop in range(n):
        next_level = set()
        for current_node in current_level:
            # 출력 이웃들
            for neighbor in graph.successors(current_node):
                if neighbor not in visited and neighbor != node:
                    next_level.add(neighbor)
            # 입력 이웃들
            for neighbor in graph.predecessors(current_node):
                if neighbor not in visited and neighbor != node:
                    next_level.add(neighbor)
        
        visited.update(current_level)
        current_level = next_level
    
    return current_level


def _extract_temporal_features(transactions: List[Dict[str, Any]], 
                             address: str, 
                             window_hours: int) -> Dict[str, float]:
    """시간 패턴 피처들을 추출합니다."""
    features = {}
    address = address.lower()
    
    # 해당 주소와 관련된 트랜잭션만 필터링
    relevant_txs = [tx for tx in transactions 
                   if tx.get('from', '').lower() == address or 
                      tx.get('to', '').lower() == address]
    
    if not relevant_txs:
        return {
            'tx_frequency_per_hour': 0.0,
            'avg_time_between_txs': 0.0,
            'time_variance': 0.0,
            'active_hours_ratio': 0.0,
            'burst_pattern_score': 0.0
        }
    
    # 타임스탬프 추출 및 정렬
    timestamps = []
    for tx in relevant_txs:
        if 'timestamp' in tx:
            timestamps.append(tx['timestamp'])
    
    if len(timestamps) < 2:
        return {
            'tx_frequency_per_hour': len(timestamps) / max(window_hours, 1),
            'avg_time_between_txs': 0.0,
            'time_variance': 0.0,
            'active_hours_ratio': 0.0,
            'burst_pattern_score': 0.0
        }
    
    timestamps.sort()
    
    # 시간당 트랜잭션 빈도
    features['tx_frequency_per_hour'] = len(timestamps) / max(window_hours, 1)
    
    # 평균 트랜잭션 간격
    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    features['avg_time_between_txs'] = np.mean(intervals)
    features['time_variance'] = np.var(intervals)
    
    # 활성 시간 비율 (시간대별 분포)
    hours = [(ts // 3600) % 24 for ts in timestamps]
    unique_hours = len(set(hours))
    features['active_hours_ratio'] = unique_hours / 24.0
    
    # 버스트 패턴 점수 (짧은 시간에 많은 거래)
    features['burst_pattern_score'] = _calculate_burst_score(timestamps)
    
    return features


def _calculate_burst_score(timestamps: List[int]) -> float:
    """버스트 패턴 점수를 계산합니다."""
    if len(timestamps) < 3:
        return 0.0
    
    # 5분 윈도우에서 3개 이상의 거래가 있는 경우를 버스트로 간주
    burst_count = 0
    window_size = 300  # 5분
    
    for i, ts in enumerate(timestamps):
        count_in_window = sum(1 for other_ts in timestamps 
                            if abs(other_ts - ts) <= window_size)
        if count_in_window >= 3:
            burst_count += 1
    
    return burst_count / len(timestamps)


def _extract_amount_features(transactions: List[Dict[str, Any]], 
                           address: str) -> Dict[str, float]:
    """금액 패턴 피처들을 추출합니다."""
    features = {}
    address = address.lower()
    
    # 입출금 분리
    incoming_amounts = []
    outgoing_amounts = []
    
    for tx in transactions:
        value = float(tx.get('value', 0))
        if tx.get('to', '').lower() == address:
            incoming_amounts.append(value)
        elif tx.get('from', '').lower() == address:
            outgoing_amounts.append(value)
    
    # 입금 패턴
    if incoming_amounts:
        features['incoming_amount_variance'] = np.var(incoming_amounts)
        features['incoming_amount_entropy'] = _calculate_amount_entropy(incoming_amounts)
        features['incoming_round_ratio'] = _calculate_round_amount_ratio(incoming_amounts)
    else:
        features['incoming_amount_variance'] = 0.0
        features['incoming_amount_entropy'] = 0.0
        features['incoming_round_ratio'] = 0.0
    
    # 출금 패턴
    if outgoing_amounts:
        features['outgoing_amount_variance'] = np.var(outgoing_amounts)
        features['outgoing_amount_entropy'] = _calculate_amount_entropy(outgoing_amounts)
        features['outgoing_round_ratio'] = _calculate_round_amount_ratio(outgoing_amounts)
    else:
        features['outgoing_amount_variance'] = 0.0
        features['outgoing_amount_entropy'] = 0.0
        features['outgoing_round_ratio'] = 0.0
    
    # 전체 금액 패턴
    all_amounts = incoming_amounts + outgoing_amounts
    if all_amounts:
        features['total_amount_variance'] = np.var(all_amounts)
        features['amount_pattern_score'] = _calculate_amount_pattern_score(all_amounts)
    else:
        features['total_amount_variance'] = 0.0
        features['amount_pattern_score'] = 0.0
    
    return features


def _calculate_amount_entropy(amounts: List[float]) -> float:
    """금액 분포의 엔트로피를 계산합니다."""
    if not amounts:
        return 0.0
    
    # 금액을 구간으로 나누어 히스토그램 생성
    bins = 10
    hist, _ = np.histogram(amounts, bins=bins)
    
    # 확률 분포 계산
    probabilities = hist / sum(hist)
    probabilities = probabilities[probabilities > 0]  # 0 제거
    
    # 엔트로피 계산
    return -sum(p * np.log2(p) for p in probabilities)


def _calculate_round_amount_ratio(amounts: List[float]) -> float:
    """정수 금액의 비율을 계산합니다."""
    if not amounts:
        return 0.0
    
    round_count = sum(1 for amount in amounts if amount == int(amount))
    return round_count / len(amounts)


def _calculate_amount_pattern_score(amounts: List[float]) -> float:
    """금액 패턴의 의심도를 계산합니다."""
    if len(amounts) < 2:
        return 0.0
    
    # 동일한 금액의 비율
    unique_amounts = len(set(amounts))
    repetition_score = 1.0 - (unique_amounts / len(amounts))
    
    # 연속된 금액의 패턴
    sorted_amounts = sorted(amounts)
    arithmetic_progression_score = _check_arithmetic_progression(sorted_amounts)
    
    return (repetition_score + arithmetic_progression_score) / 2.0


def _check_arithmetic_progression(amounts: List[float]) -> float:
    """등차수열 패턴을 확인합니다."""
    if len(amounts) < 3:
        return 0.0
    
    differences = [amounts[i+1] - amounts[i] for i in range(len(amounts)-1)]
    
    # 차이가 일정한지 확인
    if len(set(differences)) == 1:
        return 1.0
    
    # 차이의 분산이 작으면 등차수열에 가까움
    variance = np.var(differences)
    mean_diff = np.mean(differences)
    
    if mean_diff == 0:
        return 0.0
    
    coefficient_of_variation = np.sqrt(variance) / abs(mean_diff)
    return max(0.0, 1.0 - coefficient_of_variation)


def _extract_gas_features(transactions: List[Dict[str, Any]], 
                        address: str) -> Dict[str, float]:
    """가스 사용 패턴 피처들을 추출합니다."""
    features = {}
    address = address.lower()
    
    gas_prices = []
    gas_limits = []
    gas_used_ratios = []
    
    for tx in transactions:
        if (tx.get('from', '').lower() == address or 
            tx.get('to', '').lower() == address):
            
            gas_price = float(tx.get('gasPrice', 0))
            gas_limit = float(tx.get('gas', 0))
            gas_used = float(tx.get('gasUsed', gas_limit))  # gasUsed가 없으면 limit 사용
            
            if gas_price > 0:
                gas_prices.append(gas_price)
            if gas_limit > 0:
                gas_limits.append(gas_limit)
                if gas_used > 0:
                    gas_used_ratios.append(gas_used / gas_limit)
    
    # 가스 가격 패턴
    if gas_prices:
        features['gas_price_variance'] = np.var(gas_prices)
        features['gas_price_entropy'] = _calculate_amount_entropy(gas_prices)
        features['gas_price_consistency'] = 1.0 / (1.0 + np.std(gas_prices))
    else:
        features['gas_price_variance'] = 0.0
        features['gas_price_entropy'] = 0.0
        features['gas_price_consistency'] = 0.0
    
    # 가스 한도 패턴
    if gas_limits:
        features['gas_limit_variance'] = np.var(gas_limits)
        features['gas_limit_consistency'] = 1.0 / (1.0 + np.std(gas_limits))
    else:
        features['gas_limit_variance'] = 0.0
        features['gas_limit_consistency'] = 0.0
    
    # 가스 사용률 패턴
    if gas_used_ratios:
        features['gas_usage_efficiency'] = np.mean(gas_used_ratios)
        features['gas_usage_variance'] = np.var(gas_used_ratios)
    else:
        features['gas_usage_efficiency'] = 0.0
        features['gas_usage_variance'] = 0.0
    
    return features


def _extract_address_features(transactions: List[Dict[str, Any]], 
                            address: str) -> Dict[str, float]:
    """주소 패턴 피처들을 추출합니다."""
    features = {}
    address = address.lower()
    
    counterparties = set()
    contract_interactions = 0
    eoa_interactions = 0
    
    for tx in transactions:
        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()
        
        if from_addr == address:
            counterparties.add(to_addr)
            # 간단한 휴리스틱: 주소가 0으로 많이 시작하면 컨트랙트일 가능성
            if to_addr.startswith('0x000') or len(to_addr) < 42:
                contract_interactions += 1
            else:
                eoa_interactions += 1
                
        elif to_addr == address:
            counterparties.add(from_addr)
            if from_addr.startswith('0x000') or len(from_addr) < 42:
                contract_interactions += 1
            else:
                eoa_interactions += 1
    
    total_interactions = contract_interactions + eoa_interactions
    
    features['unique_counterparties'] = len(counterparties)
    features['contract_interaction_ratio'] = (contract_interactions / total_interactions 
                                            if total_interactions > 0 else 0.0)
    features['eoa_interaction_ratio'] = (eoa_interactions / total_interactions 
                                       if total_interactions > 0 else 0.0)
    
    # 주소 패턴 분석
    features['address_diversity_score'] = _calculate_address_diversity(list(counterparties))
    
    return features


def _calculate_address_diversity(addresses: List[str]) -> float:
    """주소 다양성 점수를 계산합니다."""
    if not addresses:
        return 0.0
    
    # 주소의 첫 몇 자리가 얼마나 다양한지 확인
    prefixes = [addr[:6] for addr in addresses if len(addr) >= 6]
    
    if not prefixes:
        return 0.0
    
    unique_prefixes = len(set(prefixes))
    return unique_prefixes / len(prefixes)
