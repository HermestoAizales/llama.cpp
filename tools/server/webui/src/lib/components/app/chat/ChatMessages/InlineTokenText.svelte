<script lang="ts">
	import type { TokenLogprobData } from './TokenInspector.svelte';
	import { Pin } from '@lucide/svelte';

	interface Props {
		tokens: TokenLogprobData[];
		class?: string;
	}

	let { tokens, class: className = '' }: Props = $props();

	// Popup: anchored to token element, not cursor
	let activeTokenIdx = $state<number | null>(null);
	let activeTokenRect = $state<DOMRect | null>(null);
	let popupPinned = $state(false);

	function showPopup(e: MouseEvent, idx: number) {
		if (popupPinned) return;
		const el = e.currentTarget as HTMLElement;
		activeTokenIdx = idx;
		activeTokenRect = el.getBoundingClientRect();
	}

	function hidePopup() {
		if (!popupPinned) {
			activeTokenIdx = null;
			activeTokenRect = null;
		}
	}

	function togglePin() {
		popupPinned = !popupPinned;
		if (!popupPinned) {
			activeTokenIdx = null;
			activeTokenRect = null;
		}
	}

	function formatToken(t: string): string {
		if (t === ' ') return '\u2423';
		return t.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
	}

	function computeEntropy(data: TokenLogprobData): number {
		let h = 0;
		const items =
			data.top_logprobs.length > 0
				? data.top_logprobs
				: [{ token: data.token, logprob: data.logprob }];
		for (const t of items) {
			const p = Math.exp(t.logprob);
			if (p > 0 && p <= 1) {
				h -= p * Math.log2(p);
			}
		}
		return h;
	}
</script>

<!-- Container tracks mouse for popup positioning -->
<div
	class={className}
	role="region"
	aria-label="Token colored text"
>
	{#each tokens as t, idx (t.token + idx)}
		{@const mainProb = Math.exp(t.logprob)}
		{@const ent = computeEntropy(t)}
		<span
			onmouseenter={(e) => showPopup(e, idx)}
			onmouseleave={hidePopup}
			onclick={(e) => {
				e.stopPropagation();
				showPopup(e, idx);
			}}
			class={`inline cursor-help rounded px-0.5 ${
				mainProb >= 0.5
					? 'bg-green-600/40 dark:bg-green-500/50'
					: mainProb >= 0.25
						? 'bg-yellow-600/40 dark:bg-yellow-500/50'
						: 'bg-red-600/40 dark:bg-red-500/50'
			} transition-colors ${activeTokenIdx === idx ? 'ring-2 ring-blue-400' : ''}`}>
			{t.token}
		</span>
	{/each}
</div>

<!-- Popup anchored to token element -->
{#if activeTokenIdx !== null && activeTokenRect !== null && tokens[activeTokenIdx]}
	{@const t = tokens[activeTokenIdx]}
	{@const mainProb = Math.exp(t.logprob)}
	{@const ent = computeEntropy(t)}

	<div
		class="fixed z-[9999] rounded-lg border border-border bg-card px-3.5 py-3 text-xs shadow-2xl backdrop-blur-lg"
		style="left: {activeTokenRect.left + activeTokenRect.width / 2}px; top: {activeTokenRect.bottom + 4}px; transform: translate(-50%, 0);"
		onmouseleave={() => {
			if (!popupPinned) hidePopup();
		}}
	>
		<div class="mb-2 flex items-center justify-between">
			<div class="flex items-center gap-2">
				<span class="font-mono text-base font-bold">{formatToken(t.token)}</span>
				<span
					class={`rounded px-2 py-0.5 text-xs font-semibold text-white ${
						mainProb >= 0.7 ? 'bg-green-700' : mainProb >= 0.3 ? 'bg-yellow-600' : 'bg-red-600'
					}`}>
					{(mainProb * 100).toFixed(1)}%
				</span>
			</div>
			<button
				onclick={togglePin}
				class="rounded p-1 text-muted-foreground hover:bg-muted/50 hover:text-foreground">
				<Pin class="h-3.5 w-3.5" />
				{#if popupPinned}
					<span class="text-blue-500">✓</span>
				{/if}
			</button>
		</div>
		<div class="grid grid-cols-2 gap-x-6 text-muted-foreground">
			<div>
				Logprob: <span class="text-foreground tabular-nums">{t.logprob.toFixed(3)}</span>
			</div>
			<div>
				Entropy:{' '}
				<span
					class="tabular-nums {ent < 1
						? 'text-green-500'
						: ent < 3
							? 'text-yellow-500'
							: 'text-red-500'}">
					{ent.toFixed(2)}
				</span>
			</div>
		</div>

		{#if t.top_logprobs.length > 0}
			<div class="mt-2 border-t border-border pt-2">
				<div class="mb-1 text-[10px] font-medium tracking-wider text-muted-foreground uppercase">
					Alternative Tokens
				</div>
				{#each t.top_logprobs.slice(0, 8) as alt, i (alt.token + i)}
					{@const altProb = Math.exp(alt.logprob)}
					<div class="mb-0.5 flex items-center gap-1.5 text-[11px]">
						<span class="w-4 text-right text-muted-foreground/60">#{i + 1}</span>
						<span class="w-20 truncate font-mono">{formatToken(alt.token)}</span>
						<div class="h-2.5 flex-1 overflow-hidden rounded-sm bg-muted/50">
							<div
								class={`h-full rounded-sm ${
									altProb >= 0.5
										? 'bg-green-500 dark:bg-green-600'
										: altProb >= 0.1
											? 'bg-yellow-500 dark:bg-yellow-600'
											: 'bg-red-400 dark:bg-red-500'
								}`}
								style="width: {Math.min(altProb * 100, 100)}%;"></div>
						</div>
						<span class="w-10 text-right text-muted-foreground tabular-nums">
							{(altProb * 100).toFixed(1)}%
						</span>
						<span class="w-14 text-right text-[10px] text-muted-foreground/60 tabular-nums">
							{alt.logprob.toFixed(2)}
						</span>
					</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}
